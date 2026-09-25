//! Browser live app: runs the whole pipeline on the bundled demo video (auto
//! start) or the webcam, and shows every stage as it happens: KLT tracks and
//! self-calibration while bootstrapping, then the dense model prediction,
//! tracking mask, keyframe depth maps being refined and the 3D mesh.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use dtam_core::dtam::{Keyframe, TrackStats};
use dtam_core::geom::{Intrinsics, Se3};
use dtam_core::slam::{Phase, PoseSource, Slam, SlamEvent, SlamParams};
use dtam_core::{Frame, FrameTracks, Gpu};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::capture::{Capture, sleep};
use crate::scene3d::Scene3d;

const INPUT: u32 = 512;
const DEMO_URL: &str = "demo.mp4";
const DEMO_FPS: f64 = 30.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Source {
    Demo,
    Webcam,
}

/// Resource limits, smaller on phones.
#[derive(Clone, Copy)]
struct Budget {
    mapping_max_dim: u32,
    layers: u32,
    max_keyframes: usize,
    thumb: u32,
    max_thumbs: usize,
    max_frames: usize,
}

fn is_mobile() -> bool {
    web_sys::window()
        .and_then(|w| w.navigator().user_agent().ok())
        .is_some_and(|ua| ["Mobi", "Android", "iPhone", "iPad"].iter().any(|k| ua.contains(k)))
}

impl Budget {
    fn detect() -> Self {
        if is_mobile() {
            Self { mapping_max_dim: 256, layers: 32, max_keyframes: 10, thumb: 128, max_thumbs: 600, max_frames: 3000 }
        } else {
            Self { mapping_max_dim: 512, layers: 64, max_keyframes: 32, thumb: 256, max_thumbs: 1200, max_frames: 20_000 }
        }
    }
}

/// One processed frame, shown as a unit.
struct Display {
    index: usize,
    frame: Frame,
    pose: Option<Se3>,
    klt: Option<FrameTracks>,
    stats: Option<TrackStats>,
}

/// Everything the processing task produces, read by the UI.
#[derive(Default)]
struct State {
    generation: u64,
    running: bool,
    finished: bool,
    stage: String,
    error: Option<String>,
    phase: Option<Phase>,
    frames: usize,
    fps: f32,
    /// Produced by SLAM while the current frame is being processed.
    pending_klt: Option<FrameTracks>,
    pending_stats: Option<TrackStats>,
    /// The last fully processed frame and everything computed for it. The UI
    /// shows only this, so image, pose, depth and mask are always in sync.
    display: Option<Display>,
    display_version: u64,
    keyframes: Vec<Arc<Keyframe>>,
    /// Newest keyframe emitted (possibly still being mapped).
    mapping: Option<Arc<Keyframe>>,
    mapping_version: u64,
    poses: Vec<Option<(Se3, PoseSource)>>,
    thumbs: Vec<Option<Vec<u8>>>,
    intrinsics: Option<Intrinsics>,
    self_calibrated: Option<f64>,
    mapping_k: Option<Intrinsics>,
}

impl State {
    fn apply(&mut self, ev: SlamEvent) {
        match ev {
            SlamEvent::Stage(s) => self.stage = s,
            SlamEvent::Phase(p) => self.phase = Some(p),
            SlamEvent::Klt { tracks, .. } => self.pending_klt = Some(tracks),
            SlamEvent::Intrinsics { self_calibrated, refined, mapping } => {
                self.intrinsics = Some(refined);
                self.self_calibrated = Some(self_calibrated);
                self.mapping_k = Some(mapping);
            }
            SlamEvent::Pose { frame, pose, source } => {
                if self.poses.len() <= frame {
                    self.poses.resize(frame + 1, None);
                }
                self.poses[frame] = Some((pose, source));
            }
            SlamEvent::Tracking { stats, .. } => self.pending_stats = Some(stats),
            SlamEvent::Keyframe(kf) => {
                self.mapping = Some(kf.clone());
                self.mapping_version += 1;
                // Only keyframes in the tracking model are part of `keyframes`.
                if kf.id < self.keyframes.len() {
                    let id = kf.id;
                    self.keyframes[id] = kf;
                } else if kf.id == self.keyframes.len() {
                    self.keyframes.push(kf);
                }
            }
        }
    }
}

pub struct LiveApp {
    gpu: Gpu,
    state: Rc<RefCell<State>>,
    budget: Budget,
    source: Source,
    scene: Scene3d,
    show_mask: bool,
    show_cube: bool,
    cube: Option<(u64, crate::ar::Cube)>,
    frame_tex: Option<(u64, TextureHandle)>,
    overlay_tex: Option<(u64, TextureHandle)>,
    pred_tex: Option<(u64, [TextureHandle; 2])>,
    kf_tex: Option<(u64, [TextureHandle; 3])>,
    view_frame: usize,
}

impl LiveApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let rs = cc.wgpu_render_state.as_ref().expect("WebGPU renderer");
        let gpu = Gpu::new(rs.device.clone(), rs.queue.clone());
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut app = Self {
            gpu,
            state: Rc::new(RefCell::new(State::default())),
            budget: Budget::detect(),
            source: Source::Demo,
            scene: Scene3d::default(),
            show_mask: false,
            show_cube: true,
            cube: None,
            frame_tex: None,
            overlay_tex: None,
            pred_tex: None,
            kf_tex: None,
            view_frame: 0,
        };
        app.start(Source::Demo, &cc.egui_ctx);
        app
    }

    fn params(&self) -> SlamParams {
        let mut p = SlamParams::default();
        p.mapping_max_dim = self.budget.mapping_max_dim;
        p.max_keyframes = self.budget.max_keyframes;
        p.dtam.layers = self.budget.layers;
        p
    }

    /// (Re)starts processing from `source`; any previous run stops.
    fn start(&mut self, source: Source, ctx: &egui::Context) {
        self.source = source;
        let generation = self.state.borrow().generation + 1;
        *self.state.borrow_mut() = State { generation, running: true, stage: "starting".into(), ..Default::default() };
        self.scene.recenter_next();
        self.cube = None;
        let state = self.state.clone();
        let gpu = self.gpu.clone();
        let params = self.params();
        let budget = self.budget;
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = run(state.clone(), gpu, params, budget, source, generation, ctx.clone()).await;
            let mut s = state.borrow_mut();
            if s.generation == generation {
                s.running = false;
                s.finished = true;
                if let Err(e) = result {
                    s.error = Some(e);
                }
            }
            ctx.request_repaint();
        });
    }

    fn stop(&mut self) {
        let mut s = self.state.borrow_mut();
        s.generation += 1; // the running task notices and exits
        s.running = false;
        s.finished = true;
        s.stage = "stopped".into();
    }
}

fn now_ms() -> f64 {
    web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
}

/// Per-stage wall time (ms) accumulated over frames, logged periodically.
#[derive(Default)]
struct Timing {
    seek: f64,
    grab: f64,
    thumb: f64,
    push: f64,
    yield_: f64,
    frames: u32,
}

thread_local! {
    static UI_MS: std::cell::Cell<(f64, u32)> = const { std::cell::Cell::new((0.0, 0)) };
}

/// The processing loop: grab a frame, push it through SLAM, repeat.
async fn run(
    state: Rc<RefCell<State>>,
    gpu: Gpu,
    params: SlamParams,
    budget: Budget,
    source: Source,
    generation: u64,
    ctx: egui::Context,
) -> Result<(), String> {
    let alive = |s: &Rc<RefCell<State>>| s.borrow().generation == generation;
    state.borrow_mut().stage = match source {
        Source::Demo => "loading demo video".into(),
        Source::Webcam => "waiting for camera permission".into(),
    };
    ctx.request_repaint();
    let capture = match source {
        Source::Demo => Capture::video_file(DEMO_URL, INPUT).await?,
        Source::Webcam => Capture::webcam(INPUT).await?,
    };
    let mut slam = Slam::new(gpu, (INPUT, INPUT), params);
    let demo_frames = (capture.duration() * DEMO_FPS).floor() as usize;
    let mut last_time = js_sys::Date::now();
    let mut i = 0usize;
    let mut tm = Timing::default();
    loop {
        let t0 = now_ms();
        if !alive(&state) {
            return Ok(());
        }
        let frame = match source {
            Source::Demo => {
                if i >= demo_frames {
                    break;
                }
                capture.seek((i as f64 + 0.5) / DEMO_FPS).await;
                let t1 = now_ms();
                tm.seek += t1 - t0;
                let f = capture.grab()?;
                tm.grab += now_ms() - t1;
                f
            }
            Source::Webcam => {
                if i >= budget.max_frames {
                    state.borrow_mut().stage = "frame limit reached".into();
                    break;
                }
                let f = capture.grab()?;
                tm.grab += now_ms() - t0;
                f
            }
        };
        if !alive(&state) {
            return Ok(());
        }
        let t1 = now_ms();
        let thumb = (i < budget.max_thumbs).then(|| thumbnail(&frame, budget.thumb));
        tm.thumb += now_ms() - t1;
        {
            let mut s = state.borrow_mut();
            s.pending_klt = None;
            s.pending_stats = None;
            s.thumbs.push(thumb);
            if s.poses.len() <= i {
                s.poses.resize(i + 1, None);
            }
        }
        if i % 30 == 0 {
            let mean = frame.rgb.iter().step_by(97).map(|v| *v as f64).sum::<f64>() / (frame.rgb.len() / 97) as f64;
            web_sys::console::log_1(
                &format!("frame {i}: t={:.3} mean {mean:.2} stage '{}'", capture.video.current_time(), state.borrow().stage).into(),
            );
        }
        let st = state.clone();
        let t1 = now_ms();
        slam.push(&frame, &mut |ev| {
            if let SlamEvent::Stage(s) = &ev {
                web_sys::console::log_1(&format!("stage: {s}").into());
            }
            st.borrow_mut().apply(ev)
        })
        .await;
        tm.push += now_ms() - t1;
        {
            // Commit this frame's results together.
            let mut s = state.borrow_mut();
            let pose = s.poses.get(i).copied().flatten().map(|(p, _)| p);
            let (klt, stats) = (s.pending_klt.take(), s.pending_stats.take());
            s.display = Some(Display { index: i, frame, pose, klt, stats });
            s.display_version += 1;
            s.frames = i + 1;
        }
        let now = js_sys::Date::now();
        {
            let mut s = state.borrow_mut();
            let dt = ((now - last_time) / 1000.0) as f32;
            s.fps = if s.fps == 0.0 { 1.0 / dt.max(1e-3) } else { 0.9 * s.fps + 0.1 / dt.max(1e-3) };
        }
        last_time = now;
        ctx.request_repaint();
        i += 1;
        // Let the browser paint.
        let t1 = now_ms();
        sleep(0).await;
        tm.yield_ += now_ms() - t1;
        tm.frames += 1;
        if tm.frames == 60 {
            let n = tm.frames as f64;
            let (ui_ms, ui_n) = UI_MS.with(|c| c.replace((0.0, 0)));
            web_sys::console::log_1(
                &format!(
                    "timing/frame: seek {:.1} grab {:.1} thumb {:.1} slam {:.1} yield {:.1} ms | ui {:.1} ms x {ui_n}",
                    tm.seek / n,
                    tm.grab / n,
                    tm.thumb / n,
                    tm.push / n,
                    tm.yield_ / n,
                    ui_ms / ui_n.max(1) as f64
                )
                .into(),
            );
            tm = Timing::default();
        }
    }
    if alive(&state) {
        let st = state.clone();
        slam.finish(&mut |ev| st.borrow_mut().apply(ev)).await;
        state.borrow_mut().stage = "finished — scrub the recording below".into();
    }
    Ok(())
}

/// Box-filtered (area-averaged) downscale for the recording history.
fn thumbnail(frame: &Frame, size: u32) -> Vec<u8> {
    let s = (frame.width / size).max(1);
    let mut out = Vec::with_capacity((size * size * 3) as usize);
    for y in 0..size {
        for x in 0..size {
            let mut acc = [0u32; 3];
            for dy in 0..s {
                for dx in 0..s {
                    let i = ((y * s + dy) * frame.width + x * s + dx) as usize * 3;
                    for c in 0..3 {
                        acc[c] += frame.rgb[i + c] as u32;
                    }
                }
            }
            out.extend(acc.map(|v| (v / (s * s)) as u8));
        }
    }
    out
}

fn depth_color(t: f32) -> [u8; 3] {
    if t <= 0.0 {
        return [0, 0, 0];
    }
    let t = t.clamp(0.0, 1.0);
    let r = (255.0 * (1.5 * t - 0.25).clamp(0.0, 1.0)) as u8;
    let g = (255.0 * (1.0 - (2.0 * t - 1.0).abs()).powf(0.7)) as u8;
    let b = (255.0 * (1.0 - 1.6 * t).clamp(0.0, 1.0).max(0.25)) as u8;
    [r, g, b]
}

fn depth_image(w: u32, h: u32, xi: &[f32], range: (f32, f32)) -> ColorImage {
    let rgb: Vec<u8> = xi
        .iter()
        .flat_map(|v| if *v > 0.0 { depth_color(((v - range.0) / (range.1 - range.0)).max(1e-3)) } else { [0, 0, 0] })
        .collect();
    ColorImage::from_rgb([w as usize, h as usize], &rgb)
}

fn panel(ui: &mut egui::Ui, title: &str, subtitle: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style()).fill(Color32::from_gray(22)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.strong(title);
            ui.weak(subtitle);
        });
        add(ui);
    });
}

impl eframe::App for LiveApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ui_t0 = now_ms();
        let ctx = ui.ctx().clone();
        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui, &ctx));
        let finished = self.state.borrow().finished;
        if finished {
            egui::Panel::bottom("timeline").show(ui, |ui| self.timeline(ui));
        }
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.panels(ui, frame));
        });
        if self.state.borrow().running {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let dt = now_ms() - ui_t0;
        UI_MS.with(|c| {
            let (t, n) = c.get();
            c.set((t + dt, n + 1));
        });
    }
}

impl LiveApp {
    fn top_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (running, phase, stage, frames, fps, error, nkf) = {
            let s = self.state.borrow();
            (s.running, s.phase, s.stage.clone(), s.frames, s.fps, s.error.clone(), s.keyframes.len())
        };
        ui.horizontal_wrapped(|ui| {
            ui.heading("DTAM");
            ui.weak("dense tracking and mapping · Rust + WebGPU");
            ui.separator();
            if ui.selectable_label(self.source == Source::Demo, "▶ demo video").clicked() {
                self.start(Source::Demo, ctx);
            }
            if ui.selectable_label(self.source == Source::Webcam, "📷 webcam").clicked() {
                self.start(Source::Webcam, ctx);
            }
            if running {
                if ui.button("⏹ stop").clicked() {
                    self.stop();
                }
            } else if ui.button("↻ restart").clicked() {
                self.start(self.source, ctx);
            }
        });
        ui.horizontal_wrapped(|ui| {
            let (label, color) = match phase {
                Some(Phase::Bootstrapping { collected, needed }) => {
                    (format!("BOOTSTRAP {collected}/{needed}"), Color32::from_rgb(120, 220, 120))
                }
                Some(Phase::Dense) => ("DENSE TRACKING".to_string(), Color32::from_rgb(90, 160, 255)),
                Some(Phase::Lost) => ("TRACKING LOST".to_string(), Color32::from_rgb(255, 80, 60)),
                None => ("STARTING".to_string(), Color32::GRAY),
            };
            ui.label(RichText::new(label).strong().color(Color32::BLACK).background_color(color));
            ui.label(&stage);
            ui.weak(format!("frame {frames} · {fps:.1} fps · {nkf} keyframes"));
            if let Some(k) = self.state.borrow().intrinsics {
                ui.weak(format!("f = {:.0}px ({:.1}° FOV)", k.fx, k.hfov_deg()));
            }
        });
        if phase == Some(Phase::Lost) && running {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(
                    Color32::from_rgb(255, 120, 100),
                    "Tracking failed. Point the camera back at the mapped area, or reset to start a new map.",
                );
                if ui.button("reset map").clicked() {
                    self.start(self.source, ctx);
                }
            });
        }
        if let Some(e) = error {
            ui.colored_label(Color32::from_rgb(255, 120, 100), format!("error: {e}"));
        }
    }

    fn timeline(&mut self, ui: &mut egui::Ui) {
        let n = self.state.borrow().frames;
        if n == 0 {
            return;
        }
        ui.horizontal(|ui| {
            ui.label("recording");
            let w = ui.available_width() - 80.0;
            ui.spacing_mut().slider_width = w.max(100.0);
            ui.add(egui::Slider::new(&mut self.view_frame, 0..=n - 1).text("frame"));
        });
    }

    fn panels(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let wide = ui.available_width() > 900.0;
        let finished = self.state.borrow().finished;
        let camera_sub = if finished { "recorded frame" } else { "live input, 512×512 center crop" };
        if wide {
            ui.columns(2, |cols| {
                let w = cols[0].available_width();
                panel(&mut cols[0], "Camera", camera_sub, |ui| self.camera_view(ui, (w - 16.0).min(640.0)));
                panel(&mut cols[1], "3D reconstruction", "keyframe meshes · orbit to inspect", |ui| self.scene_view(ui, frame, w - 16.0));
            });
            ui.add_space(6.0);
            ui.columns(2, |cols| {
                let w = cols[0].available_width();
                panel(&mut cols[0], "Model prediction", "model rendered at the tracked pose (what tracking aligns to)", |ui| {
                    self.prediction_view(ui, w - 16.0)
                });
                panel(&mut cols[1], "Keyframe being mapped", "reference · raw cost-volume min · regularised depth", |ui| {
                    self.keyframe_view(ui, w - 16.0)
                });
            });
        } else {
            let w = ui.available_width() - 16.0;
            panel(ui, "Camera", camera_sub, |ui| self.camera_view(ui, w.min(560.0)));
            ui.add_space(6.0);
            panel(ui, "3D reconstruction", "keyframe meshes · orbit to inspect", |ui| self.scene_view(ui, frame, w));
            ui.add_space(6.0);
            panel(ui, "Model prediction", "model rendered at the tracked pose", |ui| self.prediction_view(ui, w));
            ui.add_space(6.0);
            panel(ui, "Keyframe being mapped", "reference · raw min · regularised", |ui| self.keyframe_view(ui, w));
        }
    }

    fn scene_view(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame, width: f32) {
        let s = self.state.borrow();
        let finished = s.finished;
        let keyframes = s.keyframes.clone();
        let poses = s.poses.clone();
        // The frustum of the frame currently shown in the camera panel.
        let idx = if finished { Some(self.view_frame) } else { s.display.as_ref().map(|d| d.index) };
        let current = idx
            .and_then(|i| poses.get(i).copied().flatten())
            .zip(s.mapping_k)
            .map(|((p, _), k)| (p, k));
        drop(s);
        if let Some(rs) = frame.wgpu_render_state() {
            self.scene.ui(ui, rs, &keyframes, &poses, current, Vec2::new(width, width.min(640.0) * 0.8));
        }
        egui::CollapsingHeader::new("display options").show(ui, |ui| self.scene.options_ui(ui));
    }

    fn camera_view(&mut self, ui: &mut egui::Ui, side: f32) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_mask, "tracking mask");
            ui.checkbox(&mut self.show_cube, "AR cube");
        });
        let finished = self.state.borrow().finished;
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(side), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::BLACK);
        let s = self.state.borrow();
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let to_screen = |p: [f32; 2]| rect.min + Vec2::new((p[0] + 0.5) / INPUT as f32, (p[1] + 0.5) / INPUT as f32) * rect.width();

        // Anchor the AR cube once the first keyframe exists.
        if let (Some(k), Some(kf)) = (s.intrinsics, s.keyframes.first())
            && self.cube.as_ref().is_none_or(|c| c.0 != s.generation)
        {
            let center = [(k.width as f64 - 1.0) / 2.0, (k.height as f64 - 1.0) / 2.0];
            self.cube = crate::ar::anchor(kf, &kf.pose, &k, center, 0.12).map(|c| (s.generation, c));
        }

        // One composite image per displayed frame: the frame itself with the
        // cube rendered at that same frame's pose (occluded by that frame's
        // model depth), so they can never be out of step.
        let key = if finished { (self.view_frame as u64) << 1 | 1 } else { s.display_version << 1 };
        let key = key * 2 + self.show_cube as u64;
        if self.frame_tex.as_ref().is_none_or(|t| t.0 != key) {
            let (mut rgb, size, pose, depth) = if finished {
                let t = s.thumbs.get(self.view_frame).and_then(|t| t.clone());
                let pose = s.poses.get(self.view_frame).copied().flatten().map(|(p, _)| p);
                (t, self.budget.thumb, pose, None)
            } else {
                let d = s.display.as_ref();
                (
                    d.map(|d| d.frame.rgb.clone()),
                    INPUT,
                    d.and_then(|d| d.pose),
                    d.and_then(|d| d.stats.as_ref()).and_then(|t| t.prediction_inv_depth.as_ref()),
                )
            };
            if let Some(rgb) = rgb.as_mut() {
                if self.show_cube
                    && let (Some((_, cube)), Some(pose), Some(k)) = (&self.cube, pose, s.intrinsics)
                    && let Some((_, _, over)) =
                        dtam_core::ar::rasterize(cube, &pose, &k, size, depth.map(|(w, h, d)| (*w, *h, d.as_slice())))
                {
                    dtam_core::ar::composite(rgb, &over);
                }
                let img = ColorImage::from_rgb([size as usize, size as usize], rgb);
                self.frame_tex = Some((key, ui.ctx().load_texture("camera", img, TextureOptions::LINEAR)));
            }
        }
        if let Some((_, t)) = &self.frame_tex {
            painter.image(t.id(), rect, uv, Color32::WHITE);
        }

        let display = s.display.as_ref();
        let bootstrapping = matches!(s.phase, Some(Phase::Bootstrapping { .. }) | None);
        if !finished && bootstrapping {
            // Feature tracks while bootstrapping (they go away once dense).
            if let Some(tr) = display.and_then(|d| d.klt.as_ref()) {
                for p in &tr.points {
                    let c = if p.age == 0 { Color32::from_rgb(120, 200, 255) } else { Color32::from_rgb(40, 255, 90) };
                    painter.circle_filled(to_screen(p.pos), 2.0, c);
                }
            }
        } else if !finished && self.show_mask {
            if let Some((w, h, mask)) = display.and_then(|d| d.stats.as_ref()).and_then(|t| t.mask.as_ref()) {
                if self.overlay_tex.as_ref().is_none_or(|t| t.0 != s.display_version) {
                    let colors = [[40u8, 60, 230], [40, 230, 70], [245, 215, 40], [0, 0, 0]];
                    let rgba: Vec<u8> = mask
                        .iter()
                        .flat_map(|m| {
                            let c = colors[(*m).min(3) as usize];
                            [c[0], c[1], c[2], if *m == 3 { 0 } else { 255 }]
                        })
                        .collect();
                    let img = ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], &rgba);
                    self.overlay_tex = Some((s.display_version, ui.ctx().load_texture("mask", img, TextureOptions::NEAREST)));
                }
                if let Some((_, t)) = &self.overlay_tex {
                    painter.image(t.id(), rect, uv, Color32::WHITE.gamma_multiply(0.35));
                }
            }
        }
        if !finished && s.phase == Some(Phase::Lost) {
            painter.rect_stroke(rect, 4.0, Stroke::new(4.0, Color32::from_rgb(255, 60, 40)), egui::StrokeKind::Inside);
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "TRACKING LOST", egui::FontId::proportional(28.0), Color32::from_rgb(255, 80, 60));
        }
    }

    fn prediction_view(&mut self, ui: &mut egui::Ui, width: f32) {
        let s = self.state.borrow();
        let stats = s.display.as_ref().and_then(|d| d.stats.as_ref());
        let Some((w, h, luma, depth)) = stats.and_then(|t| t.prediction.as_ref()) else {
            ui.weak("appears once the first keyframe is mapped");
            return;
        };
        if self.pred_tex.as_ref().is_none_or(|t| t.0 != s.display_version) {
            let gray: Vec<u8> = luma.iter().flat_map(|v| [*v, *v, *v]).collect();
            let col: Vec<u8> = depth.iter().flat_map(|v| depth_color(*v as f32 / 255.0)).collect();
            let size = [*w as usize, *h as usize];
            self.pred_tex = Some((
                s.display_version,
                [
                    ui.ctx().load_texture("pred_luma", ColorImage::from_rgb(size, &gray), TextureOptions::LINEAR),
                    ui.ctx().load_texture("pred_depth", ColorImage::from_rgb(size, &col), TextureOptions::LINEAR),
                ],
            ));
        }
        let side = ((width - 20.0) / 2.0).min(300.0);
        if let Some((_, [a, b])) = &self.pred_tex {
            ui.horizontal(|ui| {
                ui.add(egui::Image::new((a.id(), Vec2::splat(side))));
                ui.add(egui::Image::new((b.id(), Vec2::splat(side))));
            });
        }
        if let Some(t) = stats {
            ui.weak(format!(
                "coverage {:.0}% · pixels used {:.0}% · rejected {:.0}% · photometric rmse {:.3} · gain {:.2}",
                t.coverage * 100.0,
                t.used_fraction * 100.0,
                t.rejected_fraction * 100.0,
                t.rmse,
                t.gain
            ));
        }
    }

    fn keyframe_view(&mut self, ui: &mut egui::Ui, width: f32) {
        let s = self.state.borrow();
        let Some(kf) = s.mapping.clone() else {
            ui.weak(match s.phase {
                Some(Phase::Bootstrapping { .. }) | None => "waiting for the feature bootstrap (move the camera sideways)",
                _ => "…",
            });
            return;
        };
        if self.kf_tex.as_ref().is_none_or(|t| t.0 != s.mapping_version) {
            let (w, h) = (kf.intrinsics.width, kf.intrinsics.height);
            let o = TextureOptions::LINEAR;
            self.kf_tex = Some((
                s.mapping_version,
                [
                    ui.ctx().load_texture("kf_rgb", ColorImage::from_rgb([w as usize, h as usize], &kf.rgb), o),
                    ui.ctx().load_texture("kf_raw", depth_image(w, h, &kf.argmin_inv_depth, kf.xi_range), o),
                    ui.ctx().load_texture("kf_reg", depth_image(w, h, &kf.inv_depth, kf.xi_range), o),
                ],
            ));
        }
        let side = ((width - 28.0) / 3.0).min(220.0);
        if let Some((_, texs)) = &self.kf_tex {
            ui.horizontal(|ui| {
                for t in texs {
                    ui.add(egui::Image::new((t.id(), Vec2::splat(side))));
                }
            });
        }
        let published = kf.id < s.keyframes.len();
        ui.weak(format!(
            "keyframe #{} (frame {}) · cost volume from {} frames · {} primal-dual iterations · {}",
            kf.id,
            kf.frame,
            kf.frames_used,
            kf.iterations,
            if published { "in the tracking model" } else { "converging" }
        ));
    }
}
