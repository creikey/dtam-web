//! egui widget around `dtam_core::viz::SceneRenderer`: keyframe meshes
//! (textured or shaded like the paper), camera path and frustums, orbit
//! controls. Rendered offscreen and shown as an egui image.

use std::sync::Arc;

use dtam_core::dtam::Keyframe;
use dtam_core::geom::{Intrinsics, Se3};
use dtam_core::slam::PoseSource;
use dtam_core::viz::{self, MeshOptions, OrbitCamera, SceneRenderer, Shading};
use eframe::egui::{self, Color32, Sense, Vec2};
use eframe::egui_wgpu;
use eframe::wgpu;

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Clone, Copy, PartialEq)]
pub struct SceneOptions {
    pub mesh: MeshOptions,
    pub shading: Shading,
    /// Show only the newest keyframe (like the paper's figures) instead of all.
    pub only_latest: bool,
    pub show_path: bool,
}

impl Default for SceneOptions {
    fn default() -> Self {
        Self {
            mesh: MeshOptions::default(),
            // The paper's texture-mapped model view.
            shading: Shading::Texture,
            only_latest: false,
            show_path: true,
        }
    }
}

type Target = (wgpu::Texture, wgpu::Texture, egui::TextureId, [u32; 2]);

pub struct Scene3d {
    renderer: Option<SceneRenderer>,
    target: Option<Target>,
    pub camera: OrbitCamera,
    pub options: SceneOptions,
    centered: bool,
}

impl Default for Scene3d {
    fn default() -> Self {
        Self { renderer: None, target: None, camera: OrbitCamera::default(), options: SceneOptions::default(), centered: false }
    }
}

impl Scene3d {
    pub fn reset_view(&mut self, keyframes: &[Arc<Keyframe>]) {
        if let Some(kf) = keyframes.first() {
            self.camera = OrbitCamera::framing(kf);
            self.centered = true;
        }
    }

    /// Forget the framing (e.g. after a pipeline reset).
    pub fn recenter_next(&mut self) {
        self.centered = false;
    }

    pub fn triangle_count(&self) -> usize {
        self.renderer.as_ref().map_or(0, |r| r.triangle_count)
    }

    /// Option widgets.
    pub fn options_ui(&mut self, ui: &mut egui::Ui) {
        let o = &mut self.options;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut o.shading, Shading::Texture, "texture");
            ui.selectable_value(&mut o.shading, Shading::TextureShaded, "texture + shading");
            ui.selectable_value(&mut o.shading, Shading::Shaded, "shaded");
        });
        ui.checkbox(&mut o.only_latest, "newest keyframe only");
        ui.checkbox(&mut o.show_path, "camera path");
        ui.checkbox(&mut o.mesh.raw_argmin, "raw arg min (unregularised)");
        ui.add(egui::Slider::new(&mut o.mesh.step, 1..=4).text("mesh step px"));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        rs: &egui_wgpu::RenderState,
        keyframes: &[Arc<Keyframe>],
        poses: &[Option<(Se3, PoseSource)>],
        current: Option<(Se3, Intrinsics)>,
        size: Vec2,
    ) {
        if !self.centered && !keyframes.is_empty() {
            self.reset_view(keyframes);
        }
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
        if resp.dragged_by(egui::PointerButton::Primary) && !ui.input(|i| i.modifiers.shift) {
            let d = resp.drag_delta();
            self.camera.rotate(d.x, d.y);
        }
        if resp.dragged_by(egui::PointerButton::Secondary)
            || (resp.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.shift))
        {
            let d = resp.drag_delta();
            self.camera.pan(d.x, d.y);
        }
        // Pinch or ctrl/cmd+scroll zooms (plain scroll keeps scrolling the page).
        if resp.hovered() {
            let zoom = ui.input(|i| i.zoom_delta());
            if zoom != 1.0 {
                self.camera.zoom(1.0 / zoom);
            }
        }
        if resp.double_clicked() {
            self.reset_view(keyframes);
        }

        let device = &rs.device;
        let renderer = self.renderer.get_or_insert_with(|| SceneRenderer::new(device, COLOR_FORMAT));
        let shown: Vec<Arc<Keyframe>> = if self.options.only_latest {
            keyframes.last().cloned().into_iter().collect()
        } else {
            keyframes.to_vec()
        };
        renderer.set_keyframes(device, &shown, self.options.mesh);
        let path: Vec<Option<Se3>> =
            if self.options.show_path { poses.iter().map(|p| p.map(|(pose, _)| pose)).collect() } else { Vec::new() };
        let lines = viz::trajectory_lines(
            &path,
            |i| match poses[i].map(|p| p.1) {
                Some(PoseSource::Bootstrap) => [120, 220, 120],
                Some(PoseSource::Dense) => [90, 160, 255],
                _ => [255, 80, 60],
            },
            keyframes,
            current.as_ref().map(|(p, k)| (*p, k)),
        );
        renderer.set_lines(device, &lines);

        let ppp = ui.ctx().pixels_per_point();
        let px = [(rect.width() * ppp).max(1.0) as u32, (rect.height() * ppp).max(1.0) as u32];
        if self.target.as_ref().is_none_or(|t| t.3 != px) {
            let tex = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("scene3d_target"),
                    size: wgpu::Extent3d { width: px[0], height: px[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let color = tex(COLOR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
            let depth = tex(viz::DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
            let view = color.create_view(&Default::default());
            let mut r = rs.renderer.write();
            let id = match self.target.take() {
                Some((_, _, id, _)) => {
                    r.update_egui_texture_from_wgpu_texture(device, &view, wgpu::FilterMode::Linear, id);
                    id
                }
                None => r.register_native_texture(device, &view, wgpu::FilterMode::Linear),
            };
            self.target = Some((color, depth, id, px));
        }
        let (color, depth, tex_id, _) = self.target.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&Default::default());
        renderer.render(
            &rs.queue,
            &mut enc,
            &color.create_view(&Default::default()),
            &depth.create_view(&Default::default()),
            px,
            &self.camera,
            self.options.shading,
            None,
        );
        rs.queue.submit([enc.finish()]);
        ui.painter().image(
            *tex_id,
            rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        ui.painter().text(
            rect.left_top() + Vec2::new(6.0, 4.0),
            egui::Align2::LEFT_TOP,
            format!("{} keyframes · {} triangles · drag orbit · shift/right-drag pan · pinch or ctrl+scroll zoom · double-click reset", keyframes.len(), self.triangle_count()),
            egui::FontId::proportional(11.0),
            Color32::from_gray(170),
        );
    }
}
