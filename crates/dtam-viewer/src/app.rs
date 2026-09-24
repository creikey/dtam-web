//! Frame-by-frame inspector for the SLAM pipeline output.

use std::sync::{Arc, Mutex};

use dtam_core::{Frame, FrameOutput, TrackedPoint};
use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Key, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Vec2,
};

const GREEN: Color32 = Color32::from_rgb(40, 255, 90);
const NEW_COLOR: Color32 = Color32::from_rgb(120, 200, 255);
const SELECTED: Color32 = Color32::from_rgb(255, 220, 40);

/// Filled in by the processing worker, read by the UI.
#[derive(Default)]
pub struct Session {
    pub source: String,
    pub frames: Vec<Frame>,
    pub outputs: Vec<FrameOutput>,
    pub expected_frames: Option<usize>,
    pub fps: f32,
    pub done: bool,
    pub error: Option<String>,
}

pub struct ViewerApp {
    session: Arc<Mutex<Session>>,
    current: usize,
    playing: bool,
    play_accum: f64,
    follow_latest: bool,

    texture: Option<TextureHandle>,
    texture_frame: Option<(usize, bool)>,

    zoom: f32,
    pan: Vec2,

    show_points: bool,
    highlight_new: bool,
    show_trails: bool,
    trail_len: usize,
    show_ids: bool,
    point_radius: f32,
    selected_track: Option<u32>,
}

impl ViewerApp {
    fn new(session: Arc<Mutex<Session>>) -> Self {
        Self {
            session,
            current: 0,
            playing: false,
            play_accum: 0.0,
            follow_latest: true,
            texture: None,
            texture_frame: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            show_points: true,
            highlight_new: true,
            show_trails: true,
            trail_len: 15,
            show_ids: false,
            point_radius: 2.5,
            selected_track: None,
        }
    }

    /// Decodes + processes the video on a worker thread using the UI's wgpu device.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_native(cc: &eframe::CreationContext<'_>, video: std::path::PathBuf, max_dim: u32) -> Self {
        let session = Arc::new(Mutex::new(Session {
            source: video.display().to_string(),
            fps: 30.0,
            ..Default::default()
        }));
        let rs = cc.wgpu_render_state.as_ref().expect("wgpu renderer required");
        let gpu = dtam_core::Gpu::new(rs.device.clone(), rs.queue.clone());
        let ctx = cc.egui_ctx.clone();
        let shared = session.clone();
        std::thread::spawn(move || {
            let result = run_pipeline(&video, max_dim, gpu, &shared, &ctx);
            let mut s = shared.lock().unwrap();
            s.done = true;
            s.error = result.err();
            ctx.request_repaint();
        });
        Self::new(session)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_pipeline(
    video: &std::path::Path,
    max_dim: u32,
    gpu: dtam_core::Gpu,
    shared: &Mutex<Session>,
    ctx: &egui::Context,
) -> Result<(), String> {
    use crate::video;
    let info = video::probe(video)?;
    let size = video::scaled_size(&info, max_dim);
    {
        let mut s = shared.lock().unwrap();
        s.expected_frames = info.frames;
        s.fps = info.fps;
    }
    let mut pipeline =
        dtam_core::SlamPipeline::new(gpu, size.0, size.1, dtam_core::TrackerParams::default());
    let t0 = std::time::Instant::now();
    video::decode(video, size, |frame| {
        let out = pollster::block_on(pipeline.process(&frame));
        let mut s = shared.lock().unwrap();
        s.frames.push(frame);
        s.outputs.push(out);
        ctx.request_repaint();
        true
    })?;
    let n = shared.lock().unwrap().frames.len();
    log::info!("processed {n} frames in {:.2}s", t0.elapsed().as_secs_f32());
    Ok(())
}

/// Points in each frame are sorted by id (survivors keep order, new ids are appended).
fn find_point(points: &[TrackedPoint], id: u32) -> Option<&TrackedPoint> {
    points.binary_search_by_key(&id, |p| p.id).ok().map(|i| &points[i])
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let session = self.session.clone();
        let s = session.lock().unwrap();
        let n = s.frames.len();

        self.handle_keys(ui.ctx(), n);
        if self.playing && n > 0 {
            let dt = ui.ctx().input(|i| i.stable_dt) as f64;
            self.play_accum += dt * s.fps as f64;
            let steps = self.play_accum.floor() as usize;
            self.play_accum -= steps as f64;
            self.current = (self.current + steps) % n;
            ui.ctx().request_repaint();
        } else if self.follow_latest && !s.done && n > 0 {
            self.current = n - 1;
        }
        self.current = self.current.min(n.saturating_sub(1));

        egui::Panel::top("top").show(ui, |ui| self.top_bar(ui, &s));
        egui::Panel::bottom("timeline").show(ui, |ui| self.timeline(ui, &s));
        egui::Panel::right("inspector")
            .default_size(260.0)
            .show(ui, |ui| self.inspector(ui, &s));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_gray(18)))
            .show(ui, |ui| self.viewport(ui, &s));
    }
}

impl ViewerApp {
    fn handle_keys(&mut self, ctx: &egui::Context, n: usize) {
        if n == 0 || ctx.egui_wants_keyboard_input() {
            return;
        }
        ctx.input(|i| {
            let step = if i.modifiers.shift { 10 } else { 1 };
            let before = self.current;
            if i.key_pressed(Key::ArrowRight) {
                self.current = (self.current + step).min(n - 1);
            }
            if i.key_pressed(Key::ArrowLeft) {
                self.current = self.current.saturating_sub(step);
            }
            if i.key_pressed(Key::Home) {
                self.current = 0;
            }
            if i.key_pressed(Key::End) {
                self.current = n - 1;
            }
            if i.key_pressed(Key::Space) {
                self.playing = !self.playing;
            }
            if self.current != before {
                self.playing = false;
                self.follow_latest = false;
            }
        });
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, s: &Session) {
        ui.horizontal(|ui| {
            ui.strong("DTAM");
            ui.separator();
            ui.label(&s.source);
            if let Some(f) = s.frames.first() {
                ui.label(format!("{}×{} @ {:.0} fps", f.width, f.height, s.fps));
            }
            ui.separator();
            let n = s.frames.len();
            if let Some(err) = &s.error {
                ui.colored_label(Color32::LIGHT_RED, err);
            } else if s.done {
                ui.label(format!("processed {n} frames"));
            } else {
                let total = s.expected_frames.unwrap_or(0);
                let frac = if total > 0 { n as f32 / total as f32 } else { 0.0 };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(240.0)
                        .text(format!("processing {n}/{}", s.expected_frames.map_or("?".into(), |t| t.to_string()))),
                );
                ui.checkbox(&mut self.follow_latest, "follow");
            }
        });
    }

    fn timeline(&mut self, ui: &mut egui::Ui, s: &Session) {
        let n = s.frames.len();
        let total = s.expected_frames.unwrap_or(n).max(n).max(1);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("⏮").clicked() {
                self.seek(0);
            }
            if ui.button("◀").clicked() {
                self.seek(self.current.saturating_sub(1));
            }
            if ui.button(if self.playing { "⏸" } else { "▶" }).clicked() {
                self.playing = !self.playing;
                self.follow_latest = false;
            }
            if ui.button("▶|").clicked() {
                self.seek((self.current + 1).min(n.saturating_sub(1)));
            }
            if ui.button("⏭").clicked() {
                self.seek(n.saturating_sub(1));
            }
            ui.monospace(format!("frame {:>4} / {}", self.current, n.saturating_sub(1)));
            ui.weak("left/right step · shift ×10 · space play · scroll zoom · drag pan · dbl-click reset");
        });

        // Bar chart of tracked points per frame; click/drag to seek.
        let (rect, resp) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), 56.0),
            Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_gray(28));
        let max_pts = s.outputs.iter().map(|o| o.tracks.points.len()).max().unwrap_or(1).max(1);
        let bar_w = rect.width() / total as f32;
        for (i, o) in s.outputs.iter().enumerate() {
            let h = rect.height() * o.tracks.points.len() as f32 / max_pts as f32;
            let x = rect.left() + i as f32 * bar_w;
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(x, rect.bottom() - h), Pos2::new(x + bar_w.max(1.0), rect.bottom())),
                0.0,
                GREEN.gamma_multiply(0.45),
            );
        }
        if n > 0 {
            let x = rect.left() + (self.current as f32 + 0.5) * bar_w;
            painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(2.0, Color32::WHITE));
        }
        painter.text(
            rect.left_top() + Vec2::new(4.0, 2.0),
            Align2::LEFT_TOP,
            format!("tracked points (max {max_pts})"),
            FontId::proportional(11.0),
            Color32::from_gray(160),
        );
        if let Some(pos) = resp.interact_pointer_pos()
            && n > 0
        {
            let i = ((pos.x - rect.left()) / bar_w).floor().max(0.0) as usize;
            self.seek(i.min(n - 1));
        }
        ui.add_space(4.0);
    }

    fn seek(&mut self, i: usize) {
        self.current = i;
        self.playing = false;
        self.follow_latest = false;
    }

    fn inspector(&mut self, ui: &mut egui::Ui, s: &Session) {
        ui.heading("Tracker");
        if let Some(out) = s.outputs.get(self.current) {
            let t = &out.tracks;
            let tracked: Vec<_> = t.points.iter().filter(|p| p.age > 0).collect();
            let mean = |f: &dyn Fn(&TrackedPoint) -> f32| {
                if tracked.is_empty() { 0.0 } else { tracked.iter().map(|p| f(p)).sum::<f32>() / tracked.len() as f32 }
            };
            egui::Grid::new("stats").num_columns(2).striped(true).show(ui, |ui| {
                ui.label("points");
                ui.monospace(t.points.len().to_string());
                ui.end_row();
                ui.label("born / lost");
                ui.monospace(format!("{} / {}", t.born, t.lost));
                ui.end_row();
                ui.label("mean age");
                ui.monospace(format!("{:.1}", mean(&|p| p.age as f32)));
                ui.end_row();
                ui.label("mean residual");
                ui.monospace(format!("{:.4}", mean(&|p| p.residual)));
                ui.end_row();
                ui.label("mean fb err");
                ui.monospace(format!("{:.3} px", mean(&|p| p.fb_error)));
                ui.end_row();
            });
        } else {
            ui.weak("waiting for frames…");
        }

        ui.separator();
        ui.heading("Display");
        ui.checkbox(&mut self.show_points, "points");
        ui.checkbox(&mut self.highlight_new, "new points in blue");
        ui.checkbox(&mut self.show_trails, "trails");
        ui.add_enabled(self.show_trails, egui::Slider::new(&mut self.trail_len, 1..=120).text("trail frames"));
        ui.checkbox(&mut self.show_ids, "track ids");
        ui.add(egui::Slider::new(&mut self.point_radius, 1.0..=8.0).text("point radius"));
        if ui.button("reset view").clicked() {
            self.zoom = 1.0;
            self.pan = Vec2::ZERO;
        }

        ui.separator();
        ui.heading("Selected track");
        match self.selected_track {
            Some(id) => {
                let hist: Vec<(usize, &TrackedPoint)> = s
                    .outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| find_point(&o.tracks.points, id).map(|p| (i, p)))
                    .collect();
                ui.monospace(format!("id {id}"));
                if let (Some(first), Some(last)) = (hist.first(), hist.last()) {
                    ui.monospace(format!("frames {}..={} ({} total)", first.0, last.0, hist.len()));
                }
                if let Some(p) = s.outputs.get(self.current).and_then(|o| find_point(&o.tracks.points, id)) {
                    ui.monospace(format!("pos ({:.2}, {:.2})", p.pos[0], p.pos[1]));
                    ui.monospace(format!("age {}  res {:.4}  fb {:.3}", p.age, p.residual, p.fb_error));
                } else {
                    ui.weak("not visible this frame");
                }
                ui.horizontal(|ui| {
                    if let Some(first) = hist.first()
                        && ui.button("go to birth").clicked()
                    {
                        self.seek(first.0);
                    }
                    if let Some(last) = hist.last()
                        && ui.button("go to death").clicked()
                    {
                        self.seek(last.0);
                    }
                    if ui.button("clear").clicked() {
                        self.selected_track = None;
                    }
                });
            }
            None => {
                ui.weak("click a point to select it");
            }
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui, s: &Session) {
        let Some(frame) = s.frames.get(self.current) else {
            ui.centered_and_justified(|ui| ui.label("decoding…"));
            return;
        };
        let (area, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());

        // Zoom around the cursor, drag to pan, double-click to reset.
        if resp.hovered() {
            let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
            let pinch = ui.ctx().input(|i| i.zoom_delta());
            let factor = (scroll * 0.002).exp() * pinch;
            if factor != 1.0
                && let Some(ptr) = resp.hover_pos()
            {
                let new_zoom = (self.zoom * factor).clamp(0.2, 40.0);
                let c = area.center() + self.pan;
                self.pan += (ptr - c) * (1.0 - new_zoom / self.zoom);
                self.zoom = new_zoom;
            }
        }
        if resp.dragged() {
            self.pan += resp.drag_delta();
        }
        if resp.double_clicked() {
            self.zoom = 1.0;
            self.pan = Vec2::ZERO;
        }

        let img_size = Vec2::new(frame.width as f32, frame.height as f32);
        let fit = (area.width() / img_size.x).min(area.height() / img_size.y);
        let scale = fit * self.zoom;
        let img_rect = Rect::from_center_size(area.center() + self.pan, img_size * scale);
        // Pixel centers are at integer coords.
        let to_screen = |p: [f32; 2]| img_rect.min + Vec2::new(p[0] + 0.5, p[1] + 0.5) * scale;

        let nearest = scale > 3.0;
        if self.texture_frame != Some((self.current, nearest)) {
            let image = ColorImage::from_rgb([frame.width as usize, frame.height as usize], &frame.rgb);
            let opts = if nearest { TextureOptions::NEAREST } else { TextureOptions::LINEAR };
            match &mut self.texture {
                Some(t) => t.set(image, opts),
                None => self.texture = Some(ui.ctx().load_texture("frame", image, opts)),
            }
            self.texture_frame = Some((self.current, nearest));
        }

        let painter = ui.painter_at(area);
        if let Some(tex) = &self.texture {
            painter.image(tex.id(), img_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }

        let Some(out) = s.outputs.get(self.current) else { return };
        let points = &out.tracks.points;

        if self.show_trails {
            let start = self.current.saturating_sub(self.trail_len);
            for p in points.iter().filter(|p| p.age > 0) {
                let mut prev = to_screen(p.pos);
                let back = (p.age as usize).min(self.current - start);
                for k in 1..=back {
                    let Some(q) = find_point(&s.outputs[self.current - k].tracks.points, p.id) else { break };
                    let cur = to_screen(q.pos);
                    let alpha = 1.0 - k as f32 / (back as f32 + 1.0);
                    painter.line_segment([prev, cur], Stroke::new(1.2, GREEN.gamma_multiply(0.8 * alpha)));
                    prev = cur;
                }
            }
        }

        if let Some(id) = self.selected_track {
            let path: Vec<Pos2> = s
                .outputs
                .iter()
                .filter_map(|o| find_point(&o.tracks.points, id).map(|p| to_screen(p.pos)))
                .collect();
            painter.add(egui::Shape::line(path, Stroke::new(1.5, SELECTED)));
        }

        if self.show_points {
            for p in points {
                let color = if self.highlight_new && p.age == 0 { NEW_COLOR } else { GREEN };
                let c = to_screen(p.pos);
                painter.circle_filled(c, self.point_radius, color);
                if Some(p.id) == self.selected_track {
                    painter.circle_stroke(c, self.point_radius + 4.0, Stroke::new(2.0, SELECTED));
                }
                if self.show_ids {
                    painter.text(
                        c + Vec2::new(self.point_radius + 2.0, 0.0),
                        Align2::LEFT_CENTER,
                        p.id.to_string(),
                        FontId::monospace(10.0),
                        color,
                    );
                }
            }
        }

        // Hover tooltip + click to select the nearest point.
        if let Some(ptr) = resp.hover_pos() {
            let hit = points
                .iter()
                .map(|p| (p, to_screen(p.pos).distance(ptr)))
                .filter(|(_, d)| *d < 10.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(p, _)| *p);
            if let Some(p) = hit {
                painter.circle_stroke(to_screen(p.pos), self.point_radius + 3.0, Stroke::new(1.5, Color32::WHITE));
                resp.clone().on_hover_ui_at_pointer(|ui| {
                    ui.monospace(format!(
                        "id {}\npos ({:.2}, {:.2})\nage {}\nresidual {:.4}\nfb err {:.3} px",
                        p.id, p.pos[0], p.pos[1], p.age, p.residual, p.fb_error
                    ));
                });
            }
            if resp.clicked() {
                self.selected_track = hit.map(|p| p.id);
            }
        }
    }
}
