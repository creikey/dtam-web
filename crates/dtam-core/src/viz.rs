//! Offscreen 3D rendering of the dense model: every keyframe's inverse depth
//! map as a triangle mesh (textured with its reference image, or shaded like
//! the paper's figures), plus camera trajectory / frustum lines.
//! Used by the viewers (native + web) and by the CLI for headless images.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Point3, Vector3};
use wgpu::util::DeviceExt;

use crate::dtam::Keyframe;
use crate::geom::{Intrinsics, Se3};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const SHADER: &str = r#"
struct Cam {
    view_proj: mat4x4f,
    eye: vec4f,
    misc: vec4f, // shading mode (0 texture, 1 shaded, 2 both), _, _, _
}
@group(0) @binding(0) var<uniform> cam: Cam;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
    @location(1) normal: vec3f,
    @location(2) world: vec3f,
}

fn to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@vertex
fn vs_mesh(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) c: vec4f) -> VOut {
    var out: VOut;
    out.pos = cam.view_proj * vec4f(p, 1.0);
    out.color = to_linear(c.rgb);
    out.normal = n;
    out.world = p;
    return out;
}

@fragment
fn fs_mesh(in: VOut) -> @location(0) vec4f {
    let v = normalize(cam.eye.xyz - in.world);
    var n = normalize(in.normal);
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    // Headlight Phong, like the paper's shaded reconstructions.
    let diffuse = max(dot(n, v), 0.0);
    let spec = pow(max(dot(reflect(-v, n), v), 0.0), 24.0);
    let shade = 0.12 + 0.75 * diffuse + 0.35 * spec;
    let mode = u32(cam.misc.x);
    var c = in.color;
    if (mode == 1u) {
        c = vec3f(shade);
    } else if (mode == 2u) {
        c = in.color * (0.35 + 0.8 * diffuse) + vec3f(0.25 * spec);
    }
    return vec4f(c, 1.0);
}

@vertex
fn vs_lines(@location(0) p: vec3f, @location(1) c: vec4f) -> VOut {
    var out: VOut;
    out.pos = cam.view_proj * vec4f(p, 1.0);
    out.color = to_linear(c.rgb);
    out.normal = vec3f(0.0, 0.0, 1.0);
    out.world = p;
    return out;
}

@fragment
fn fs_lines(in: VOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshVertex {
    pos: [f32; 3],
    normal: [f32; 3],
    color: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CamUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    misc: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shading {
    Texture,
    Shaded,
    TextureShaded,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshOptions {
    /// Pixel step of the mesh grid (1 = full keyframe resolution).
    pub step: usize,
    /// Triangles whose normal is more oblique than this to the keyframe ray
    /// are culled (they bridge depth discontinuities).
    pub max_oblique_deg: f32,
    /// Triangles with a vertex below this depth confidence are culled.
    pub min_confidence: f32,
    /// Show the raw arg min instead of the regularised depth.
    pub raw_argmin: bool,
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self { step: 2, max_oblique_deg: 80.0, min_confidence: 0.0, raw_argmin: false }
    }
}

/// Orbit camera looking at `target`.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y_deg: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: Vector3::new(0.0, 0.0, 1.0), yaw: 0.0, pitch: 0.25, distance: 1.8, fov_y_deg: 50.0 }
    }
}

impl OrbitCamera {
    /// Frames the first keyframe: looking at its median depth from behind it.
    pub fn framing(kf: &Keyframe) -> Self {
        let depth = 1.0 / kf.median_inv_depth.max(1e-3) as f64;
        let t = kf.pose.transform(&Vector3::new(0.0, 0.0, depth));
        let fwd = kf.pose.r * Vector3::new(0.0, 0.0, 1.0);
        // Look along the keyframe's viewing direction: -eye_dir = fwd.
        let pitch = (fwd.y as f32).clamp(-1.0, 1.0).asin();
        let yaw = (-(fwd.x as f32)).atan2(fwd.z as f32);
        Self {
            target: Vector3::new(t.x as f32, t.y as f32, t.z as f32),
            yaw,
            pitch: pitch + 0.15,
            distance: (depth * 1.3) as f32,
            fov_y_deg: 50.0,
        }
    }

    pub fn eye(&self) -> Vector3<f32> {
        let dir = Vector3::new(self.yaw.sin() * self.pitch.cos(), -self.pitch.sin(), -self.yaw.cos() * self.pitch.cos());
        self.target + dir * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Matrix4<f32> {
        let eye = self.eye();
        // World is camera convention (y down), so "up" is -y.
        let view = Matrix4::look_at_rh(&Point3::from(eye), &Point3::from(self.target), &Vector3::new(0.0, -1.0, 0.0));
        let (near, far) = (0.005 * self.distance.max(0.01), 1000.0);
        let f = 1.0 / (self.fov_y_deg.to_radians() / 2.0).tan();
        #[rustfmt::skip]
        let proj = Matrix4::new(
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, far / (near - far), near * far / (near - far),
            0.0, 0.0, -1.0, 0.0,
        );
        proj * view
    }

    /// Drag to orbit.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    /// Drag to pan in the view plane.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = Vector3::new(self.yaw.cos(), 0.0, self.yaw.sin());
        let up = Vector3::new(self.yaw.sin() * self.pitch.sin(), self.pitch.cos(), -self.yaw.cos() * self.pitch.sin());
        let s = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * s;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.01, 500.0);
    }
}

struct GpuMesh {
    key: (usize, usize, u32),
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
}

pub struct SceneRenderer {
    mesh_pipe: wgpu::RenderPipeline,
    lines_pipe: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    meshes: HashMap<usize, GpuMesh>,
    options: Option<MeshOptions>,
    lines: Option<(wgpu::Buffer, u32)>,
    pub triangle_count: usize,
}

impl SceneRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mesh_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4];
        let line_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4];
        let make = |vs: &str, fs: &str, stride: usize, attrs: &[wgpu::VertexAttribute], topology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(vs),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: attrs,
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
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(color_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_cam"),
            size: size_of::<CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        Self {
            mesh_pipe: make("vs_mesh", "fs_mesh", size_of::<MeshVertex>(), &mesh_attrs, wgpu::PrimitiveTopology::TriangleList),
            lines_pipe: make("vs_lines", "fs_lines", size_of::<LineVertex>(), &line_attrs, wgpu::PrimitiveTopology::LineList),
            uniform,
            bind,
            meshes: HashMap::new(),
            options: None,
            lines: None,
            triangle_count: 0,
        }
    }

    /// Uploads meshes for keyframes that are new or changed since last call.
    pub fn set_keyframes(&mut self, device: &wgpu::Device, keyframes: &[Arc<Keyframe>], opts: MeshOptions) {
        if self.options != Some(opts) {
            self.meshes.clear();
            self.options = Some(opts);
        }
        self.meshes.retain(|id, _| *id < keyframes.len());
        for kf in keyframes {
            let key = (kf.id, kf.frames_used, kf.iterations);
            if self.meshes.get(&kf.id).is_some_and(|m| m.key == key) {
                continue;
            }
            let (verts, idx) = build_mesh(kf, &opts);
            if idx.is_empty() {
                self.meshes.remove(&kf.id);
                continue;
            }
            let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("kf_mesh_vertices"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("kf_mesh_indices"),
                contents: bytemuck::cast_slice(&idx),
                usage: wgpu::BufferUsages::INDEX,
            });
            self.meshes.insert(kf.id, GpuMesh { key, vertices, indices, count: idx.len() as u32 });
        }
        self.triangle_count = self.meshes.values().map(|m| m.count as usize / 3).sum();
    }

    pub fn set_lines(&mut self, device: &wgpu::Device, lines: &[LineVertex]) {
        self.lines = (!lines.is_empty()).then(|| {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scene_lines"),
                contents: bytemuck::cast_slice(lines),
                usage: wgpu::BufferUsages::VERTEX,
            });
            (buf, lines.len() as u32)
        });
    }

    /// Draws into `color` / `depth` (same size), clearing both.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        size: [u32; 2],
        camera: &OrbitCamera,
        shading: Shading,
        only_keyframe: Option<usize>,
    ) {
        let eye = camera.eye();
        let cu = CamUniform {
            view_proj: camera.view_proj(size[0] as f32 / size[1].max(1) as f32).into(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            misc: [
                match shading {
                    Shading::Texture => 0.0,
                    Shading::Shaded => 1.0,
                    Shading::TextureShaded => 2.0,
                },
                0.0,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&cu));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_pipeline(&self.mesh_pipe);
        for (id, m) in &self.meshes {
            if only_keyframe.is_some_and(|k| k != *id) {
                continue;
            }
            pass.set_vertex_buffer(0, m.vertices.slice(..));
            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.count, 0, 0..1);
        }
        if let Some((buf, n)) = &self.lines {
            pass.set_pipeline(&self.lines_pipe);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..*n, 0..1);
        }
    }
}

fn build_mesh(kf: &Keyframe, opts: &MeshOptions) -> (Vec<MeshVertex>, Vec<u32>) {
    let k = &kf.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let step = opts.step.max(1);
    let (gw, gh) = (w.div_ceil(step), h.div_ceil(step));
    let raw = if opts.raw_argmin { &kf.argmin_inv_depth } else { &kf.inv_depth };
    // Display-only 3x3 median: removes isolated single-pixel depth spikes
    // (the model used for tracking is untouched).
    let depth: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            let mut n = [0f32; 9];
            let mut c = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = ((x + dx).clamp(0, w as i64 - 1), (y + dy).clamp(0, h as i64 - 1));
                    n[c] = raw[(yy as usize) * w + xx as usize];
                    c += 1;
                }
            }
            n.sort_by(f32::total_cmp);
            n[4]
        })
        .collect();
    // Keyframe-frame points on the grid.
    let mut pts: Vec<Option<Vector3<f64>>> = Vec::with_capacity(gw * gh);
    for gy in 0..gh {
        for gx in 0..gw {
            let (x, y) = ((gx * step).min(w - 1), (gy * step).min(h - 1));
            let i = y * w + x;
            let xi = depth[i];
            let ok = xi > 0.0 && kf.confidence[i] >= opts.min_confidence;
            pts.push(ok.then(|| k.unproject([x as f64, y as f64]) / xi as f64));
        }
    }
    let min_cos = opts.max_oblique_deg.to_radians().cos() as f64;
    let max_jump = 0.02 * step as f64;
    let mut normals = vec![Vector3::<f64>::zeros(); gw * gh];
    let mut idx = Vec::new();
    let tri = |a: usize, b: usize, c: usize, normals: &mut Vec<Vector3<f64>>, idx: &mut Vec<u32>| {
        let (Some(pa), Some(pb), Some(pc)) = (pts[a], pts[b], pts[c]) else { return };
        let n = (pb - pa).cross(&(pc - pa));
        let len = n.norm();
        if len <= 0.0 {
            return;
        }
        let center = (pa + pb + pc) / 3.0;
        // Oblique AND spanning a real depth jump: bridges a discontinuity.
        // (Pixel-scale depth noise can tilt a tiny triangle without any jump.)
        let (za, zb, zc) = (pa.z, pb.z, pc.z);
        let jump = za.max(zb).max(zc) / za.min(zb).min(zc) - 1.0;
        if (n.dot(&center) / (len * center.norm())).abs() < min_cos && jump > max_jump {
            return;
        }
        for v in [a, b, c] {
            normals[v] += n;
        }
        idx.extend_from_slice(&[a as u32, b as u32, c as u32]);
    };
    for gy in 0..gh - 1 {
        for gx in 0..gw - 1 {
            let i = gy * gw + gx;
            tri(i, i + 1, i + gw, &mut normals, &mut idx);
            tri(i + 1, i + gw + 1, i + gw, &mut normals, &mut idx);
        }
    }
    let verts = (0..gw * gh)
        .map(|i| {
            let (gx, gy) = (i % gw, i / gw);
            let (x, y) = ((gx * step).min(w - 1), (gy * step).min(h - 1));
            let p = pts[i].map(|p| kf.pose.transform(&p)).unwrap_or_default();
            let n = kf.pose.r * normals[i];
            let n = if n.norm() > 0.0 { n.normalize() } else { n };
            let c = &kf.rgb[(y * w + x) * 3..][..3];
            MeshVertex {
                pos: [p.x as f32, p.y as f32, p.z as f32],
                normal: [n.x as f32, n.y as f32, n.z as f32],
                color: [c[0], c[1], c[2], 255],
            }
        })
        .collect();
    (verts, idx)
}

/// Camera path (colored by `color_of`) and frustums, as line vertices.
pub fn trajectory_lines(
    poses: &[Option<Se3>],
    color_of: impl Fn(usize) -> [u8; 3],
    keyframes: &[Arc<Keyframe>],
    current: Option<(Se3, &Intrinsics)>,
) -> Vec<LineVertex> {
    let mut lines = Vec::new();
    let v = |p: &Vector3<f64>, c: [u8; 3]| LineVertex { pos: [p.x as f32, p.y as f32, p.z as f32], color: [c[0], c[1], c[2], 255] };
    for i in 1..poses.len() {
        if let (Some(a), Some(b)) = (poses[i - 1], poses[i]) {
            let c = color_of(i);
            lines.push(v(&a.t, c));
            lines.push(v(&b.t, c));
        }
    }
    let scale = keyframes.first().map_or(0.08, |k| 0.08 / k.median_inv_depth.max(1e-3) as f64);
    let mut frustum = |pose: &Se3, k: &Intrinsics, size: f64, c: [u8; 3]| {
        let corners = [[0.0, 0.0], [k.width as f64 - 1.0, 0.0], [k.width as f64 - 1.0, k.height as f64 - 1.0], [0.0, k.height as f64 - 1.0]]
            .map(|u| pose.transform(&(k.unproject(u) * size)));
        for i in 0..4 {
            lines.extend([v(&pose.t, c), v(&corners[i], c), v(&corners[i], c), v(&corners[(i + 1) % 4], c)]);
        }
    };
    for kf in keyframes {
        frustum(&kf.pose, &kf.intrinsics, scale, [255, 170, 40]);
    }
    if let Some((pose, k)) = current {
        frustum(&pose, k, scale * 1.4, [255, 255, 255]);
    }
    lines
}

/// Renders a still image of the model (headless). Returns tightly packed RGBA.
pub async fn render_image(
    gpu: &crate::Gpu,
    keyframes: &[Arc<Keyframe>],
    lines: &[LineVertex],
    camera: &OrbitCamera,
    size: [u32; 2],
    shading: Shading,
    opts: MeshOptions,
) -> Vec<u8> {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = SceneRenderer::new(&gpu.device, format);
    r.set_keyframes(&gpu.device, keyframes, opts);
    r.set_lines(&gpu.device, lines);
    let tex = |format, usage| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = tex(format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let depth = tex(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let row = (size[0] * 4).next_multiple_of(256);
    let rb = gpu.readback("scene_rb", (row * size[1]) as u64);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    r.render(&gpu.queue, &mut enc, &color.create_view(&Default::default()), &depth.create_view(&Default::default()), size, camera, shading, None);
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size[1]) },
        },
        wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    let data = gpu.read_buffers(&[(&rb, (row * size[1]) as u64)]).await.remove(0);
    let mut out = Vec::with_capacity((size[0] * size[1] * 4) as usize);
    for y in 0..size[1] as usize {
        out.extend_from_slice(&data[y * row as usize..][..size[0] as usize * 4]);
    }
    out
}
