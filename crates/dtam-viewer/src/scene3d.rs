//! 3D view of the fused model: every keyframe's inverse depth map
//! back-projected to colored world points, plus the camera trajectory and
//! frustums. Rendered offscreen with wgpu and shown as an egui image.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use dtam_core::dtam::Keyframe;
use dtam_core::geom::{Intrinsics, Se3};
use dtam_core::slam::PoseSource;
use eframe::egui::{self, Color32, Sense, Vec2};
use eframe::egui_wgpu;
use eframe::wgpu;
use nalgebra::{Matrix4, Point3, Vector3};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const SHADER: &str = r#"
struct Cam {
    view_proj: mat4x4f,
    viewport: vec4f, // width px, height px, point size px, _
}
@group(0) @binding(0) var<uniform> cam: Cam;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
}

fn to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@vertex
fn vs_points(@builtin(vertex_index) vi: u32, @location(0) p: vec3f, @location(1) c: vec4f) -> VOut {
    var corners = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
        vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    var clip = cam.view_proj * vec4f(p, 1.0);
    clip = vec4f(clip.xy + corners[vi] * cam.viewport.z / cam.viewport.xy * clip.w, clip.zw);
    var out: VOut;
    out.pos = clip;
    out.color = to_linear(c.rgb);
    return out;
}

@vertex
fn vs_lines(@location(0) p: vec3f, @location(1) c: vec4f) -> VOut {
    var out: VOut;
    out.pos = cam.view_proj * vec4f(p, 1.0);
    out.color = to_linear(c.rgb);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CamUniform {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 4],
}

/// What the scene draws, as chosen in the UI.
#[derive(Clone, Copy, PartialEq)]
pub struct SceneOptions {
    pub point_step: usize,
    pub min_confidence: f32,
    pub raw_argmin: bool,
    pub color_by_keyframe: bool,
    pub point_size: f32,
    pub show_frustums: bool,
    pub show_path: bool,
}

impl Default for SceneOptions {
    fn default() -> Self {
        Self {
            point_step: 2,
            min_confidence: 0.3,
            raw_argmin: false,
            color_by_keyframe: false,
            point_size: 1.5,
            show_frustums: true,
            show_path: true,
        }
    }
}

struct Gpu {
    points_pipe: wgpu::RenderPipeline,
    lines_pipe: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    target: Option<(wgpu::Texture, wgpu::Texture, egui::TextureId, [u32; 2])>,
    points: Option<(wgpu::Buffer, u32)>,
    lines: Option<(wgpu::Buffer, u32)>,
}

pub struct Scene3d {
    gpu: Option<Gpu>,
    built_for: Option<(usize, usize, SceneOptions)>,
    pub options: SceneOptions,
    target: Vector3<f32>,
    yaw: f32,
    pitch: f32,
    distance: f32,
    pub point_count: usize,
}

impl Default for Scene3d {
    fn default() -> Self {
        Self {
            gpu: None,
            built_for: None,
            options: SceneOptions::default(),
            target: Vector3::new(0.0, 0.0, 1.0),
            yaw: 0.0,
            pitch: 0.25,
            distance: 1.8,
            point_count: 0,
        }
    }
}

fn pipelines(device: &wgpu::Device) -> Gpu {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scene3d"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene3d"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scene3d"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4];
    let make = |entry: &str, step: wgpu::VertexStepMode, topology: wgpu::PrimitiveTopology| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: step,
                    attributes: &attrs,
                })],
            },
            primitive: wgpu::PrimitiveState { topology, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(COLOR_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene3d_cam"),
        size: size_of::<CamUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene3d"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
    });
    Gpu {
        points_pipe: make("vs_points", wgpu::VertexStepMode::Instance, wgpu::PrimitiveTopology::TriangleList),
        lines_pipe: make("vs_lines", wgpu::VertexStepMode::Vertex, wgpu::PrimitiveTopology::LineList),
        uniform,
        bind,
        target: None,
        points: None,
        lines: None,
    }
}

fn keyframe_color(id: usize) -> [u8; 3] {
    let h = (id as f32 * 0.618_034).fract() * 6.0;
    let x = (1.0 - (h % 2.0 - 1.0).abs()) * 200.0;
    let (r, g, b) = match h as u32 {
        0 => (200.0, x, 40.0),
        1 => (x, 200.0, 40.0),
        2 => (40.0, 200.0, x),
        3 => (40.0, x, 200.0),
        4 => (x, 40.0, 200.0),
        _ => (200.0, 40.0, x),
    };
    [r as u8, g as u8, b as u8]
}

fn upload(device: &wgpu::Device, verts: &[Vertex]) -> Option<(wgpu::Buffer, u32)> {
    if verts.is_empty() {
        return None;
    }
    use wgpu::util::DeviceExt;
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene3d_vertices"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    Some((buf, verts.len() as u32))
}

fn frustum(lines: &mut Vec<Vertex>, pose: &Se3, k: &Intrinsics, size: f64, color: [u8; 3]) {
    let c = pose.t;
    let corners = [[0.0, 0.0], [k.width as f64 - 1.0, 0.0], [k.width as f64 - 1.0, k.height as f64 - 1.0], [0.0, k.height as f64 - 1.0]]
        .map(|u| pose.transform(&(k.unproject(u) * size)));
    let v = |p: &Vector3<f64>| Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], color: [color[0], color[1], color[2], 255] };
    for i in 0..4 {
        lines.push(v(&c));
        lines.push(v(&corners[i]));
        lines.push(v(&corners[i]));
        lines.push(v(&corners[(i + 1) % 4]));
    }
}

impl Scene3d {
    /// Orbit camera: centers the view on the first keyframe's median depth.
    pub fn reset_view(&mut self, keyframes: &[Arc<Keyframe>]) {
        *self = Self { gpu: self.gpu.take(), built_for: None, options: self.options, ..Default::default() };
        if let Some(kf) = keyframes.first() {
            let depth = 1.0 / kf.median_inv_depth.max(1e-3) as f64;
            let t = kf.pose.transform(&Vector3::new(0.0, 0.0, depth));
            self.target = Vector3::new(t.x as f32, t.y as f32, t.z as f32);
            self.distance = (depth * 1.6) as f32;
        }
    }

    fn view_proj(&self, aspect: f32) -> Matrix4<f32> {
        let dir = Vector3::new(
            self.yaw.sin() * self.pitch.cos(),
            -self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        let eye = self.target + dir * self.distance;
        // World is camera convention (y down), so "up" is -y.
        let view = Matrix4::look_at_rh(&Point3::from(eye), &Point3::from(self.target), &Vector3::new(0.0, -1.0, 0.0));
        let (near, far) = (0.005 * self.distance.max(0.01), 1000.0);
        let f = 1.0 / (50f32.to_radians() / 2.0).tan();
        #[rustfmt::skip]
        let proj = Matrix4::new(
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, far / (near - far), near * far / (near - far),
            0.0, 0.0, -1.0, 0.0,
        );
        proj * view
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        rs: &egui_wgpu::RenderState,
        keyframes: &[Arc<Keyframe>],
        poses: &[Option<(Se3, PoseSource)>],
        intrinsics: Option<Intrinsics>,
        current: usize,
    ) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if resp.dragged_by(egui::PointerButton::Primary) && !ui.input(|i| i.modifiers.shift) {
            let d = resp.drag_delta();
            self.yaw += d.x * 0.008;
            self.pitch = (self.pitch + d.y * 0.008).clamp(-1.55, 1.55);
        }
        if resp.dragged_by(egui::PointerButton::Secondary)
            || (resp.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.shift))
        {
            let d = resp.drag_delta();
            let right = Vector3::new(self.yaw.cos(), 0.0, self.yaw.sin());
            let up = Vector3::new(self.yaw.sin() * self.pitch.sin(), self.pitch.cos(), -self.yaw.cos() * self.pitch.sin());
            let s = self.distance * 0.0015;
            self.target += (-right * d.x + up * d.y) * s;
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            let zoom = ui.input(|i| i.zoom_delta());
            self.distance = (self.distance * (-scroll * 0.002).exp() / zoom).clamp(0.01, 500.0);
        }
        if resp.double_clicked() {
            self.reset_view(keyframes);
        }

        let ppp = ui.ctx().pixels_per_point();
        let size = [(rect.width() * ppp).max(1.0) as u32, (rect.height() * ppp).max(1.0) as u32];
        let vp = self.view_proj(size[0] as f32 / size[1] as f32);
        let device = &rs.device;
        let gpu = self.gpu.get_or_insert_with(|| pipelines(device));

        // Points: rebuilt when the keyframe set or options change.
        let key = (keyframes.len(), keyframes.iter().map(|k| k.frames_used).sum::<usize>(), self.options);
        if self.built_for != Some(key) {
            let mut verts = Vec::new();
            for kf in keyframes {
                let (w, h) = (kf.intrinsics.width as usize, kf.intrinsics.height as usize);
                let depth = if self.options.raw_argmin { &kf.argmin_inv_depth } else { &kf.inv_depth };
                let tint = keyframe_color(kf.id);
                for y in (0..h).step_by(self.options.point_step.max(1)) {
                    for x in (0..w).step_by(self.options.point_step.max(1)) {
                        let i = y * w + x;
                        let xi = depth[i];
                        if xi <= 0.0 || kf.confidence[i] < self.options.min_confidence {
                            continue;
                        }
                        let p = kf.pose.transform(&(kf.intrinsics.unproject([x as f64, y as f64]) / xi as f64));
                        let c = if self.options.color_by_keyframe { tint } else { [kf.rgb[i * 3], kf.rgb[i * 3 + 1], kf.rgb[i * 3 + 2]] };
                        verts.push(Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], color: [c[0], c[1], c[2], 255] });
                    }
                }
            }
            self.point_count = verts.len();
            gpu.points = upload(device, &verts);
            self.built_for = Some(key);
        }

        // Lines: trajectory + frustums, rebuilt every frame (cheap).
        let mut lines = Vec::new();
        let scale = keyframes.first().map_or(0.08, |k| 0.08 / k.median_inv_depth.max(1e-3) as f64);
        if self.options.show_path {
            for pair in poses.windows(2) {
                if let [Some((a, _)), Some((b, src))] = pair {
                    let c = match src {
                        PoseSource::Bootstrap => [120, 220, 120],
                        PoseSource::Dense => [90, 160, 255],
                        PoseSource::Predicted => [255, 80, 60],
                    };
                    for p in [a.t, b.t] {
                        lines.push(Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], color: [c[0], c[1], c[2], 255] });
                    }
                }
            }
        }
        if let Some(k) = intrinsics {
            if self.options.show_frustums {
                for kf in keyframes {
                    frustum(&mut lines, &kf.pose, &k, scale, [255, 170, 40]);
                }
            }
            if let Some(Some((pose, _))) = poses.get(current) {
                frustum(&mut lines, pose, &k, scale * 1.3, [255, 255, 255]);
            }
        }
        gpu.lines = upload(device, &lines);

        // Offscreen target at physical resolution.
        if gpu.target.as_ref().is_none_or(|t| t.3 != size) {
            let tex = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("scene3d_target"),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let color = tex(COLOR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
            let depth = tex(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
            let view = color.create_view(&Default::default());
            let mut renderer = rs.renderer.write();
            let id = match gpu.target.take() {
                Some((_, _, id, _)) => {
                    renderer.update_egui_texture_from_wgpu_texture(device, &view, wgpu::FilterMode::Linear, id);
                    id
                }
                None => renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear),
            };
            gpu.target = Some((color, depth, id, size));
        }
        let (color, depth, tex_id, _) = gpu.target.as_ref().unwrap();

        let cu = CamUniform {
            view_proj: vp.into(),
            viewport: [size[0] as f32, size[1] as f32, self.options.point_size * ppp, 0.0],
        };
        rs.queue.write_buffer(&gpu.uniform, 0, bytemuck::bytes_of(&cu));
        let cview = color.create_view(&Default::default());
        let dview = depth.create_view(&Default::default());
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cview,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.012, g: 0.014, b: 0.018, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &dview,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &gpu.bind, &[]);
            if let Some((buf, n)) = &gpu.points {
                pass.set_pipeline(&gpu.points_pipe);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..*n);
            }
            if let Some((buf, n)) = &gpu.lines {
                pass.set_pipeline(&gpu.lines_pipe);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..*n, 0..1);
            }
        }
        rs.queue.submit([enc.finish()]);

        ui.painter().image(
            *tex_id,
            rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        ui.painter().text(
            rect.left_top() + Vec2::new(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            format!(
                "{} keyframes · {} points · drag orbit · shift/right-drag pan · scroll zoom · double-click reset",
                keyframes.len(),
                self.point_count
            ),
            egui::FontId::proportional(12.0),
            Color32::from_gray(170),
        );
    }
}
