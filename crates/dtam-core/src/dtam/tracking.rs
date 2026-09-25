//! Dense tracking (paper §2.3): model prediction by rendering all keyframe
//! meshes into a virtual camera, rotation pre-alignment between consecutive
//! live frames, then coarse-to-fine 6DOF forward-compositional alignment of
//! the live image to the prediction. Normal equations are reduced on the GPU.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::{DtamParams, Keyframe};
use crate::geom::{Intrinsics, Mat3, Se3, Vec3};
use crate::gpu::{Gpu, dispatch_2d};
use crate::Frame;

const NWG: u32 = 128;
const NACC: u64 = 36;
const NEAR: f32 = 0.01;
const FAR: f32 = 1000.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PyrParams {
    src_off: u32,
    src_w: u32,
    src_h: u32,
    mode: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PredictParams {
    t: [[f32; 4]; 4],
    k_r: [f32; 4],
    k_v: [f32; 4],
    dims: [u32; 4],
    misc: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TrackParams {
    t: [[f32; 4]; 4],
    k: [f32; 4],
    dims: [u32; 4],
    misc: [f32; 4],
}

/// Mirror of `GnState` in track_types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GnState {
    cand: [[f32; 4]; 4],
    pose: [[f32; 4]; 4],
    sums: [f32; 32],
    sums_hi: [f32; 4],
    best: f32,
    damping: f32,
    gain: f32,
    bias: f32,
    done: u32,
    has_acc: u32,
    iters: u32,
    _pad: u32,
}

impl GnState {
    fn start(t: &Se3, (gain, bias): (f32, f32)) -> Self {
        let m = t.to_mat4_f32();
        Self { cand: m, pose: m, gain, bias, damping: 1e-4, ..Self::zeroed() }
    }

    fn sums(&self) -> [f64; NACC as usize] {
        let mut out = [0f64; NACC as usize];
        for (o, v) in out.iter_mut().zip(self.sums.iter().chain(&self.sums_hi)) {
            *o = *v as f64;
        }
        out
    }
}

fn se3_from_mat4(m: &[[f32; 4]; 4]) -> Se3 {
    let r = Mat3::from_fn(|i, j| m[j][i] as f64);
    Se3::new(crate::geom::orthonormalize(&r), Vec3::new(m[3][0] as f64, m[3][1] as f64, m[3][2] as f64))
}

#[derive(Clone, Copy)]
struct Level {
    off: u32,
    w: u32,
    h: u32,
}

/// The model rendered into a virtual camera.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// `T_wv` of the virtual camera.
    pub pose: Se3,
    /// Fraction of pixels with predicted surface.
    pub coverage: f32,
    /// Percentiles (2, 50, 98) of predicted inverse depth; 0 if no surface.
    pub inv_depth_percentiles: [f32; 3],
    /// Predicted luma and inverse depth at pyramid level 1 (for display).
    pub preview: Option<(u32, u32, Vec<f32>, Vec<f32>)>,
}

#[derive(Clone, Debug, Default)]
pub struct TrackStats {
    pub rmse: f32,
    /// Pixels used / pixels with a prediction in view (finest level).
    pub used_fraction: f32,
    pub rejected_fraction: f32,
    pub coverage: f32,
    pub iterations: u32,
    /// Photometric compensation a I_l + b applied to the live image.
    pub gain: f32,
    pub bias: f32,
    /// Per-pixel class at pyramid level 1: 0 no model, 1 used, 2 rejected, 3 out of view.
    pub mask: Option<(u32, u32, Vec<u8>)>,
    /// Predicted luma and inverse depth (scaled to 0..255 over the 2-98th
    /// percentile) at pyramid level 1, for display.
    pub prediction: Option<(u32, u32, Vec<u8>, Vec<u8>)>,
    /// Predicted inverse depth at pyramid level 1, as ξ × 2000 in u16 (0 = no
    /// surface), for occlusion tests.
    pub prediction_inv_depth: Option<(u32, u32, Vec<u16>)>,
}

impl Prediction {
    /// Display / occlusion products of a prediction, without tracking.
    pub fn to_stats(&self) -> TrackStats {
        let mut stats = TrackStats { coverage: self.coverage, ..Default::default() };
        stats.fill_prediction(self);
        stats
    }
}

impl TrackStats {
    fn fill_prediction(&mut self, pred: &Prediction) {
        let Some((w, h, luma, depth)) = &pred.preview else { return };
        let [lo, _, hi] = pred.inv_depth_percentiles;
        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
        self.prediction = Some((
            *w,
            *h,
            luma.iter().map(|v| to8(*v)).collect(),
            depth.iter().map(|v| if *v > 0.0 { to8((v - lo) / (hi - lo).max(1e-6)).max(1) } else { 0 }).collect(),
        ));
        self.prediction_inv_depth =
            Some((*w, *h, depth.iter().map(|v| (v * 2000.0).round().clamp(0.0, 65535.0) as u16).collect()));
    }
}

struct ModelKf {
    kf: Arc<Keyframe>,
    _depth: wgpu::Buffer,
    _rgb: wgpu::Buffer,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    index_count: u32,
}

pub struct DenseTracker {
    k: Intrinsics,
    min_view_cos: f32,
    levels: Vec<Level>,
    pyr_len: u32,

    unpack: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    track6: wgpu::ComputePipeline,
    track_rot: wgpu::ComputePipeline,
    gn_begin: wgpu::ComputePipeline,
    gn_step6: wgpu::ComputePipeline,
    gn_step_rot: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,

    live_rgba: wgpu::Buffer,
    _live_pyr: wgpu::Buffer,
    pred_luma: wgpu::Buffer,
    pred_depth: wgpu::Buffer,
    partials: wgpu::Buffer,
    partials_rb: wgpu::Buffer,
    mask: wgpu::Buffer,
    mask_rb: wgpu::Buffer,
    preview_rb: wgpu::Buffer,
    track_params: wgpu::Buffer,

    live_bgs: [(wgpu::BindGroup, Vec<wgpu::BindGroup>); 2],
    pred_down_bgs: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    track6_bg: wgpu::BindGroup,
    gn_state: wgpu::Buffer,
    gn_rb: wgpu::Buffer,
    gn_begin_bg: wgpu::BindGroup,
    /// Per pyramid level: (uniform, residual pass bind group, solver bind group)
    /// for the 6DOF alignment and for the rotation pre-alignment.
    level6: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup)>,
    level_rot: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup)>,
    mask_params: wgpu::Buffer,
    mask_bg: wgpu::BindGroup,

    luma_tex: wgpu::Texture,
    depth_tex: wgpu::Texture,
    zbuf: wgpu::Texture,
    index_buf: Option<(wgpu::Buffer, u32, u32, u32)>,

    model: Vec<ModelKf>,
    /// Last good (gain, bias), used to start the next frame.
    photometric: (f32, f32),
}

impl DenseTracker {
    pub fn new(gpu: &Gpu, k: Intrinsics, params: &DtamParams) -> Self {
        let mut levels = vec![Level { off: 0, w: k.width, h: k.height }];
        for l in 1..params.track_levels.max(1) {
            let last = *levels.last().unwrap();
            let kl = k.level(l);
            levels.push(Level { off: last.off + last.w * last.h, w: kl.width, h: kl.height });
        }
        let last = *levels.last().unwrap();
        let pyr_len = last.off + last.w * last.h;
        let n0 = (k.width * k.height) as u64;

        let pyr_src = include_str!("shaders/pyr.wgsl");
        let common = include_str!("shaders/track_common.wgsl");
        let unpack = gpu.compute_pipeline_entry("unpack", pyr_src, "unpack");
        let down = gpu.compute_pipeline_entry("down", pyr_src, "down");
        let types = include_str!("shaders/track_types.wgsl");
        let track6 = gpu.compute_pipeline("track6", &[types, common, include_str!("shaders/track6.wgsl")].concat());
        let track_rot = gpu.compute_pipeline("track_rot", &[types, common, include_str!("shaders/track_rot.wgsl")].concat());
        let gn_src = [types, include_str!("shaders/gn.wgsl")].concat();
        let gn_begin = gpu.compute_pipeline_entry("gn_begin", &gn_src, "begin");
        let gn_step6 = gpu.compute_pipeline_entry("gn_step6", &gn_src, "step6");
        let gn_step_rot = gpu.compute_pipeline_entry("gn_step_rot", &gn_src, "step_rot");

        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("predict"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/predict.wgsl").into()),
        });
        let target = Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R32Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        });
        let render = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("predict"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
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
                targets: &[target.clone(), target],
            }),
            multiview_mask: None,
            cache: None,
        });

        let rw = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let live_rgba = gpu.storage("live_rgba", n0 * 4, rw);
        let live_pyr = gpu.storage("live_pyr", 2 * pyr_len as u64 * 4, rw);
        let pred_luma = gpu.storage("pred_luma", pyr_len as u64 * 4, rw);
        let pred_depth = gpu.storage("pred_depth", pyr_len as u64 * 4, rw);
        let partials = gpu.storage("partials", NWG as u64 * NACC * 4, rw);
        let partials_rb = gpu.readback("partials_rb", NWG as u64 * NACC * 4);
        let mask = gpu.storage("mask", n0 * 4, rw);
        let mask_rb = gpu.readback("mask_rb", n0 * 4);
        let preview_rb = gpu.readback("preview_rb", n0 * 8);
        let track_params = gpu.uniform("track_params", &TrackParams::zeroed());

        let pyr_uniform = |src: Level, dst: Level, mode: u32| {
            gpu.uniform(
                "pyr_params",
                &PyrParams {
                    src_off: src.off,
                    src_w: src.w,
                    src_h: src.h,
                    mode,
                    dst_off: dst.off,
                    dst_w: dst.w,
                    dst_h: dst.h,
                    _pad: 0,
                },
            )
        };
        let shift = |l: Level, base: u32| Level { off: l.off + base, ..l };
        let live_bgs = [0u32, 1].map(|slot| {
            let base = slot * pyr_len;
            let u = pyr_uniform(levels[0], shift(levels[0], base), 0);
            let unpack_bg = bind_at(gpu, &unpack, &[(0, &u), (1, &live_rgba), (2, &live_pyr)]);
            let downs = levels
                .windows(2)
                .map(|p| {
                    let u = pyr_uniform(shift(p[0], base), shift(p[1], base), 0);
                    bind_at(gpu, &down, &[(0, &u), (2, &live_pyr)])
                })
                .collect();
            (unpack_bg, downs)
        });
        let pred_down_bgs = levels
            .windows(2)
            .map(|p| {
                let ul = pyr_uniform(p[0], p[1], 0);
                let ud = pyr_uniform(p[0], p[1], 1);
                (
                    bind_at(gpu, &down, &[(0, &ul), (2, &pred_luma)]),
                    bind_at(gpu, &down, &[(0, &ud), (2, &pred_depth)]),
                )
            })
            .collect();
        let gn_state = gpu.storage("gn_state", std::mem::size_of::<GnState>() as u64, rw);
        let gn_rb = gpu.readback("gn_rb", std::mem::size_of::<GnState>() as u64);
        let track6_bind = |u: &wgpu::Buffer| {
            bind_at(
                gpu,
                &track6,
                &[(0, u), (1, &live_pyr), (2, &pred_luma), (3, &pred_depth), (4, &partials), (5, &mask), (6, &gn_state)],
            )
        };
        let track6_bg = track6_bind(&track_params);
        let gn_begin_bg = bind_at(gpu, &gn_begin, &[(6, &gn_state)]);
        let level6 = levels
            .iter()
            .map(|_| {
                let u = gpu.uniform("track_level", &TrackParams::zeroed());
                let (a, b) = (track6_bind(&u), bind_at(gpu, &gn_step6, &[(0, &u), (4, &partials), (6, &gn_state)]));
                (u, a, b)
            })
            .collect();
        let level_rot = levels
            .iter()
            .map(|_| {
                let u = gpu.uniform("rot_level", &TrackParams::zeroed());
                let a = bind_at(gpu, &track_rot, &[(0, &u), (1, &live_pyr), (2, &partials), (6, &gn_state)]);
                let b = bind_at(gpu, &gn_step_rot, &[(4, &partials), (6, &gn_state)]);
                (u, a, b)
            })
            .collect();
        let mask_params = gpu.uniform("mask_params", &TrackParams::zeroed());
        let mask_bg = track6_bind(&mask_params);

        let tex = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: k.width, height: k.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        Self {
            k,
            min_view_cos: params.max_oblique_deg.to_radians().cos(),
            levels,
            pyr_len,
            unpack,
            down,
            track6,
            track_rot,
            gn_begin,
            gn_step6,
            gn_step_rot,
            render,
            live_rgba,
            _live_pyr: live_pyr,
            pred_luma,
            pred_depth,
            partials,
            partials_rb,
            mask,
            mask_rb,
            preview_rb,
            track_params,
            live_bgs,
            pred_down_bgs,
            track6_bg,
            gn_state,
            gn_rb,
            gn_begin_bg,
            level6,
            level_rot,
            mask_params,
            mask_bg,
            luma_tex: tex("pred_luma_tex", wgpu::TextureFormat::R32Float, color_usage),
            depth_tex: tex("pred_depth_tex", wgpu::TextureFormat::R32Float, color_usage),
            zbuf: tex("pred_z", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT),
            index_buf: None,
            model: Vec::new(),
            photometric: (1.0, 0.0),
        }
    }

    pub fn add_model_keyframe(&mut self, gpu: &Gpu, kf: Arc<Keyframe>, depth: wgpu::Buffer, rgb: wgpu::Buffer) {
        let (w, h) = (kf.intrinsics.width, kf.intrinsics.height);
        if self.index_buf.as_ref().is_none_or(|b| (b.1, b.2) != (w, h)) {
            let mut idx = Vec::with_capacity(((w - 1) * (h - 1) * 6) as usize);
            for y in 0..h - 1 {
                for x in 0..w - 1 {
                    let i = y * w + x;
                    idx.extend_from_slice(&[i, i + 1, i + w, i + 1, i + w + 1, i + w]);
                }
            }
            let count = idx.len() as u32;
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grid_indices"),
                size: (idx.len() * 4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&idx));
            self.index_buf = Some((buf, w, h, count));
        }
        let uniform = gpu.uniform("predict_params", &PredictParams::zeroed());
        let layout = self.render.get_bind_group_layout(0);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("predict"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: depth.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rgb.as_entire_binding() },
            ],
        });
        let index_count = self.index_buf.as_ref().unwrap().3;
        self.model.push(ModelKf { kf, _depth: depth, _rgb: rgb, uniform, bind, index_count });
    }

    pub fn update_model_keyframe(&mut self, id: usize, kf: Arc<Keyframe>) {
        if let Some(m) = self.model.get_mut(id) {
            m.kf = kf;
        }
    }

    pub fn set_live(&mut self, gpu: &Gpu, frame: &Frame, slot: usize) {
        assert_eq!((frame.width, frame.height), (self.k.width, self.k.height), "live frame size");
        gpu.queue.write_buffer(&self.live_rgba, 0, bytemuck::cast_slice(&frame.pack_rgba()));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            let (unpack_bg, downs) = &self.live_bgs[slot];
            pass.set_pipeline(&self.unpack);
            pass.set_bind_group(0, unpack_bg, &[]);
            dispatch_2d(&mut pass, self.levels[0].w, self.levels[0].h);
            pass.set_pipeline(&self.down);
            for (bg, lvl) in downs.iter().zip(&self.levels[1..]) {
                pass.set_bind_group(0, bg, &[]);
                dispatch_2d(&mut pass, lvl.w, lvl.h);
            }
        }
        gpu.queue.submit([enc.finish()]);
    }

    /// Renders every keyframe into the virtual camera `t_wv`.
    pub async fn predict(&mut self, gpu: &Gpu, t_wv: Se3) -> Prediction {
        self.submit_predict(gpu, t_wv);
        let data = gpu.read_buffers(&[(&self.preview_rb, self.preview_bytes())]).await;
        self.finish_predict(t_wv, &data[0])
    }

    fn preview_bytes(&self) -> u64 {
        let l1 = self.levels[1.min(self.levels.len() - 1)];
        (l1.w * l1.h * 8) as u64
    }

    /// Renders the model at `t_wv` and queues the level-1 preview copy into
    /// `preview_rb` (read back by the caller).
    fn submit_predict(&mut self, gpu: &Gpu, t_wv: Se3) {
        let k = self.k;
        let t_vw = t_wv.inverse();
        for m in &self.model {
            let t_vr = t_vw.compose(&m.kf.pose);
            let pp = PredictParams {
                t: t_vr.to_mat4_f32(),
                k_r: m.kf.intrinsics.to_f32(),
                k_v: k.to_f32(),
                dims: [m.kf.intrinsics.width, m.kf.intrinsics.height, k.width, k.height],
                misc: [NEAR, FAR, self.min_view_cos, 0.0],
            };
            gpu.queue.write_buffer(&m.uniform, 0, bytemuck::bytes_of(&pp));
        }
        let luma_view = self.luma_tex.create_view(&Default::default());
        let depth_view = self.depth_tex.create_view(&Default::default());
        let z_view = self.zbuf.create_view(&Default::default());
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("predict"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &luma_view, depth_slice: None, resolve_target: None, ops: clear }),
                    Some(wgpu::RenderPassColorAttachment { view: &depth_view, depth_slice: None, resolve_target: None, ops: clear }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &z_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if let Some((ibuf, ..)) = &self.index_buf {
                pass.set_pipeline(&self.render);
                pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for m in &self.model {
                    pass.set_bind_group(0, &m.bind, &[]);
                    pass.draw_indexed(0..m.index_count, 0, 0..1);
                }
            }
        }
        let extent = wgpu::Extent3d { width: k.width, height: k.height, depth_or_array_layers: 1 };
        for (tex, buf) in [(&self.luma_tex, &self.pred_luma), (&self.depth_tex, &self.pred_depth)] {
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(k.width * 4), rows_per_image: Some(k.height) },
                },
                extent,
            );
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.down);
            for ((bl, bd), lvl) in self.pred_down_bgs.iter().zip(&self.levels[1..]) {
                pass.set_bind_group(0, bl, &[]);
                dispatch_2d(&mut pass, lvl.w, lvl.h);
                pass.set_bind_group(0, bd, &[]);
                dispatch_2d(&mut pass, lvl.w, lvl.h);
            }
        }
        // Level 1 luma + depth for stats and display.
        let l1 = self.levels[1.min(self.levels.len() - 1)];
        let bytes = (l1.w * l1.h * 4) as u64;
        enc.copy_buffer_to_buffer(&self.pred_luma, l1.off as u64 * 4, &self.preview_rb, 0, bytes);
        enc.copy_buffer_to_buffer(&self.pred_depth, l1.off as u64 * 4, &self.preview_rb, bytes, bytes);
        gpu.queue.submit([enc.finish()]);
    }

    fn finish_predict(&self, t_wv: Se3, preview: &[u8]) -> Prediction {
        let l1 = self.levels[1.min(self.levels.len() - 1)];
        let all: Vec<f32> = bytemuck::pod_collect_to_vec(preview);
        let (luma, depth) = all.split_at(all.len() / 2);
        let mut valid: Vec<f32> = depth.iter().copied().filter(|v| *v > 0.0).collect();
        valid.sort_by(f32::total_cmp);
        let pct = |p: f32| valid.get(((valid.len().saturating_sub(1)) as f32 * p) as usize).copied().unwrap_or(0.0);
        Prediction {
            pose: t_wv,
            coverage: valid.len() as f32 / depth.len() as f32,
            inv_depth_percentiles: [pct(0.02), pct(0.5), pct(0.98)],
            preview: Some((l1.w, l1.h, luma.to_vec(), depth.to_vec())),
        }
    }

    async fn run_pass(&self, gpu: &Gpu, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, tp: &TrackParams) -> [f64; NACC as usize] {
        gpu.queue.write_buffer(&self.track_params, 0, bytemuck::bytes_of(tp));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(NWG, 1, 1);
        }
        let size = NWG as u64 * NACC * 4;
        enc.copy_buffer_to_buffer(&self.partials, 0, &self.partials_rb, 0, size);
        gpu.queue.submit([enc.finish()]);
        let data = gpu.read_buffers(&[(&self.partials_rb, size)]).await;
        let parts: Vec<f32> = bytemuck::pod_collect_to_vec(&data[0]);
        let mut sum = [0f64; NACC as usize];
        for chunk in parts.chunks_exact(NACC as usize) {
            for (s, v) in sum.iter_mut().zip(chunk) {
                *s += *v as f64;
            }
        }
        sum
    }

    /// Rotation-only alignment between consecutive live frames (§2.3.1).
    /// All Gauss-Newton iterations run on the GPU; one readback.
    pub async fn rotation(&mut self, gpu: &Gpu, params: &DtamParams, prev_slot: usize, cur_slot: usize) -> Mat3 {
        gpu.queue.write_buffer(&self.gn_state, 0, bytemuck::bytes_of(&GnState::start(&Se3::identity(), (1.0, 0.0))));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for &l in &params.rotation_levels {
                let Some(lvl) = self.levels.get(l as usize).copied() else { continue };
                let (u, track_bg, step_bg) = &self.level_rot[l as usize];
                let tp = TrackParams {
                    t: Se3::identity().to_mat4_f32(),
                    k: self.k.level(l).to_f32(),
                    dims: [lvl.w, lvl.h, cur_slot as u32 * self.pyr_len + lvl.off, prev_slot as u32 * self.pyr_len + lvl.off],
                    misc: [0.2, 0.0, 0.0, 0.0],
                };
                gpu.queue.write_buffer(u, 0, bytemuck::bytes_of(&tp));
                pass.set_pipeline(&self.gn_begin);
                pass.set_bind_group(0, &self.gn_begin_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                for _ in 0..params.rotation_iterations {
                    pass.set_pipeline(&self.track_rot);
                    pass.set_bind_group(0, track_bg, &[]);
                    pass.dispatch_workgroups(NWG, 1, 1);
                    pass.set_pipeline(&self.gn_step_rot);
                    pass.set_bind_group(0, step_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
        }
        let size = std::mem::size_of::<GnState>() as u64;
        enc.copy_buffer_to_buffer(&self.gn_state, 0, &self.gn_rb, 0, size);
        gpu.queue.submit([enc.finish()]);
        let data = gpu.read_buffers(&[(&self.gn_rb, size)]).await;
        let st: GnState = bytemuck::pod_read_unaligned(&data[0]);
        se3_from_mat4(&st.pose).r
    }

    /// Robust photometric cost (truncated quadratic, level 0) of the live frame
    /// in `slot` against `pred` if the live camera were at `t_wl`.
    /// Returns (cost, used fraction).
    pub async fn cost(&mut self, gpu: &Gpu, params: &DtamParams, slot: usize, pred: &Prediction, t_wl: Se3) -> (f64, f64) {
        let lvl = self.levels[0];
        let thresh = params.track_thresholds.last().copied().unwrap_or(0.1);
        let t_lv = t_wl.inverse().compose(&pred.pose);
        let (gain, bias) = self.photometric;
        let tp = TrackParams {
            t: t_lv.to_mat4_f32(),
            k: self.k.to_f32(),
            dims: [lvl.w, lvl.h, slot as u32 * self.pyr_len + lvl.off, lvl.off],
            misc: [thresh, 0.0, gain, bias],
        };
        let s = self.run_pass(gpu, &self.track6, &self.track6_bg, &tp).await;
        let t2 = (thresh as f64).powi(2);
        ((s[27] + s[28] * t2) / s[29].max(1.0), s[30] / s[29].max(1.0))
    }

    /// Coarse-to-fine 6DOF alignment of the live frame in `slot` against the
    /// prediction; returns `T_wl`.
    pub async fn align(
        &mut self,
        gpu: &Gpu,
        params: &DtamParams,
        slot: usize,
        pred: &Prediction,
        want_mask: bool,
    ) -> (Se3, TrackStats) {
        self.submit_align(gpu, params, slot, want_mask);
        let (bufs, sizes) = self.align_readbacks(want_mask);
        let reqs: Vec<_> = bufs.iter().zip(&sizes).map(|(b, s)| (*b, *s)).collect();
        let data = gpu.read_buffers(&reqs).await;
        self.finish_align(pred, &data[0], data.get(1).map(|v| v.as_slice()))
    }

    /// Renders the prediction at `t_wv` and aligns the live frame in `slot`
    /// to it, with a single GPU->CPU readback for both.
    pub async fn predict_align(
        &mut self,
        gpu: &Gpu,
        params: &DtamParams,
        slot: usize,
        t_wv: Se3,
        want_mask: bool,
    ) -> (Prediction, Se3, TrackStats) {
        self.submit_predict(gpu, t_wv);
        self.submit_align(gpu, params, slot, want_mask);
        let (mut bufs, mut sizes) = self.align_readbacks(want_mask);
        bufs.push(&self.preview_rb);
        sizes.push(self.preview_bytes());
        let reqs: Vec<_> = bufs.iter().zip(&sizes).map(|(b, s)| (*b, *s)).collect();
        let data = gpu.read_buffers(&reqs).await;
        let pred = self.finish_predict(t_wv, data.last().unwrap());
        let (pose, stats) = self.finish_align(&pred, &data[0], if want_mask { Some(&data[1]) } else { None });
        (pred, pose, stats)
    }

    fn align_readbacks(&self, want_mask: bool) -> (Vec<&wgpu::Buffer>, Vec<u64>) {
        let mut bufs = vec![&self.gn_rb];
        let mut sizes = vec![std::mem::size_of::<GnState>() as u64];
        if want_mask {
            let l = self.levels[1.min(self.levels.len() - 1)];
            bufs.push(&self.mask_rb);
            sizes.push((l.w * l.h * 4) as u64);
        }
        (bufs, sizes)
    }

    /// Queues the whole coarse-to-fine alignment: per level, a fixed number
    /// of (residual pass, solver step) pairs; the solver skips the remaining
    /// iterations of a level once it has converged, like the CPU loop's
    /// early exits.
    fn submit_align(&mut self, gpu: &Gpu, params: &DtamParams, slot: usize, want_mask: bool) {
        gpu.queue.write_buffer(&self.gn_state, 0, bytemuck::bytes_of(&GnState::start(&Se3::identity(), self.photometric)));
        let nlev = self.levels.len();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (li, l) in (0..nlev).rev().enumerate() {
                let lvl = self.levels[l];
                let (u, track_bg, step_bg) = &self.level6[l];
                let tp = TrackParams {
                    t: Se3::identity().to_mat4_f32(),
                    k: self.k.level(l as u32).to_f32(),
                    dims: [lvl.w, lvl.h, slot as u32 * self.pyr_len + lvl.off, lvl.off],
                    misc: [params.track_thresholds.get(li).copied().unwrap_or(0.1), 0.0, 1.0, 0.0],
                };
                gpu.queue.write_buffer(u, 0, bytemuck::bytes_of(&tp));
                pass.set_pipeline(&self.gn_begin);
                pass.set_bind_group(0, &self.gn_begin_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                for _ in 0..params.track_iterations.get(li).copied().unwrap_or(5) {
                    pass.set_pipeline(&self.track6);
                    pass.set_bind_group(0, track_bg, &[]);
                    pass.dispatch_workgroups(NWG, 1, 1);
                    pass.set_pipeline(&self.gn_step6);
                    pass.set_bind_group(0, step_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            if want_mask && nlev > 1 {
                let lvl = self.levels[1];
                let tp = TrackParams {
                    t: Se3::identity().to_mat4_f32(),
                    k: self.k.level(1).to_f32(),
                    dims: [lvl.w, lvl.h, slot as u32 * self.pyr_len + lvl.off, lvl.off],
                    misc: [params.track_thresholds.last().copied().unwrap_or(0.1), 1.0, 1.0, 0.0],
                };
                gpu.queue.write_buffer(&self.mask_params, 0, bytemuck::bytes_of(&tp));
                pass.set_pipeline(&self.track6);
                pass.set_bind_group(0, &self.mask_bg, &[]);
                pass.dispatch_workgroups(NWG, 1, 1);
            }
        }
        let size = std::mem::size_of::<GnState>() as u64;
        enc.copy_buffer_to_buffer(&self.gn_state, 0, &self.gn_rb, 0, size);
        if want_mask && nlev > 1 {
            let l = self.levels[1];
            enc.copy_buffer_to_buffer(&self.mask, 0, &self.mask_rb, 0, (l.w * l.h * 4) as u64);
        }
        gpu.queue.submit([enc.finish()]);
    }

    fn finish_align(&mut self, pred: &Prediction, state: &[u8], mask: Option<&[u8]>) -> (Se3, TrackStats) {
        let st: GnState = bytemuck::pod_read_unaligned(state);
        let last = st.sums();
        let mut stats = TrackStats { coverage: pred.coverage, iterations: st.iters, ..Default::default() };
        if mask.is_some() {
            stats.fill_prediction(pred);
        }
        if last[29] > 0.0 {
            stats.rmse = (last[27] / last[30].max(1.0)).sqrt() as f32;
            stats.used_fraction = (last[30] / last[29]) as f32;
            stats.rejected_fraction = (last[28] / last[29]) as f32;
        }
        stats.gain = st.gain;
        stats.bias = st.bias;
        if let (Some(m), true) = (mask, self.levels.len() > 1) {
            let l = self.levels[1];
            let m: Vec<u32> = bytemuck::pod_collect_to_vec(m);
            stats.mask = Some((l.w, l.h, m.into_iter().map(|v| v as u8).collect()));
        }
        if stats.used_fraction > 0.5 {
            self.photometric = (st.gain, st.bias);
        }
        let t_lv = se3_from_mat4(&st.pose);
        (pred.pose.compose(&t_lv.inverse()), stats)
    }
}

fn bind_at(gpu: &Gpu, pipe: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })
        .collect();
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    })
}
