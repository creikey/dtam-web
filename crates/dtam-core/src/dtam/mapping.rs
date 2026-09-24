//! Keyframe construction: photometric cost volume over many frames, then the
//! non-convex regularised solve of paper §2.2.3 (primal-dual Huber-TV coupled
//! to an exhaustive, accelerated point-wise search with a Newton step).

use bytemuck::{Pod, Zeroable};

use super::{DtamParams, Keyframe};
use crate::geom::{Intrinsics, Se3};
use crate::gpu::{Gpu, dispatch_2d};
use crate::Frame;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MapParams {
    w: u32,
    h: u32,
    layers: u32,
    min_count: u32,
    xi_min: f32,
    xi_step: f32,
    theta: f32,
    lambda: f32,
    eps: f32,
    sigma_q: f32,
    sigma_d: f32,
    alpha: f32,
    beta: f32,
    _p1: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameXf {
    m: [[f32; 4]; 3],
    b: [f32; 4],
}

pub struct Mapper {
    cost_update: wgpu::ComputePipeline,
    minmax: wgpu::ComputePipeline,
    weights: wgpu::ComputePipeline,
    dual: wgpu::ComputePipeline,
    primal: wgpu::ComputePipeline,
    aux: wgpu::ComputePipeline,
    confidence: wgpu::ComputePipeline,
}

/// A keyframe whose cost volume is still live on the GPU (paper §2.2.1: "keeping
/// the average cost up to date as each overlapping frame from I(r) arrives").
/// Frames can keep being added and the regularised solution re-solved.
pub struct ActiveKeyframe {
    pub frame: usize,
    pub t_wr: Se3,
    k: Intrinsics,
    reference_rgb: Vec<u8>,
    xi_range: (f32, f32),
    lambda: f32,
    mp: MapParams,
    n: u64,
    params_buf: wgpu::Buffer,
    xf_buf: wgpu::Buffer,
    img_buf: wgpu::Buffer,
    _vol_sum: wgpu::Buffer,
    _vol_cnt: wgpu::Buffer,
    stats: wgpu::Buffer,
    pub(super) depth: wgpu::Buffer,
    pub(super) rgb: wgpu::Buffer,
    update_bg: wgpu::BindGroup,
    minmax_bg: wgpu::BindGroup,
    weights_bg: wgpu::BindGroup,
    dual_bg: wgpu::BindGroup,
    primal_bg: wgpu::BindGroup,
    aux_bg: wgpu::BindGroup,
    conf_bg: wgpu::BindGroup,
    conf: wgpu::Buffer,
    q: wgpu::Buffer,
    pub frames_used: usize,
    pub frames_since_solve: usize,
    max_dist: f64,
}

fn shader(parts: &[&str]) -> String {
    parts.concat()
}

impl Mapper {
    pub fn new(gpu: &Gpu) -> Self {
        let common = include_str!("shaders/map_common.wgsl");
        let access = include_str!("shaders/cost_access.wgsl");
        Self {
            cost_update: gpu.compute_pipeline("cost_update", &shader(&[common, include_str!("shaders/cost_update.wgsl")])),
            minmax: gpu.compute_pipeline("cost_minmax", &shader(&[common, include_str!("shaders/cost_minmax.wgsl"), access])),
            weights: gpu.compute_pipeline("weights", &shader(&[common, include_str!("shaders/weights.wgsl")])),
            dual: gpu.compute_pipeline("dual", &shader(&[common, include_str!("shaders/dual.wgsl")])),
            primal: gpu.compute_pipeline("primal", &shader(&[common, include_str!("shaders/primal.wgsl")])),
            aux: gpu.compute_pipeline("aux", &shader(&[common, include_str!("shaders/aux.wgsl"), access])),
            confidence: gpu.compute_pipeline("confidence", &shader(&[common, include_str!("shaders/confidence.wgsl"), access])),
        }
    }

    /// Starts a keyframe at `reference` / `t_wr` with an empty cost volume.
    pub fn begin(
        &self,
        gpu: &Gpu,
        k: &Intrinsics,
        params: &DtamParams,
        frame: usize,
        reference: &Frame,
        t_wr: Se3,
        xi_range: (f32, f32),
        lambda: f32,
    ) -> ActiveKeyframe {
        let (w, h) = (k.width, k.height);
        assert_eq!((reference.width, reference.height), (w, h), "reference size must match intrinsics");
        let layers = params.layers.next_multiple_of(4);
        let n = (w * h) as u64;
        let (xi_min, xi_max) = xi_range;
        let mp = MapParams {
            w,
            h,
            layers,
            min_count: params.min_voxel_views,
            xi_min,
            xi_step: (xi_max - xi_min) / (layers - 1) as f32,
            theta: params.theta_start,
            lambda,
            eps: params.epsilon,
            sigma_q: params.sigma_q,
            sigma_d: params.sigma_d,
            alpha: params.alpha,
            beta: params.beta,
            _p1: [0.0; 3],
        };
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let params_buf = gpu.uniform("map_params", &mp);
        let xf_buf = gpu.uniform("frame_xf", &FrameXf::zeroed());
        let rgb = gpu.storage_init("ref_rgba", bytemuck::cast_slice(&reference.pack_rgba()), usage);
        let img_buf = gpu.storage("frame_rgba", n * 4, usage);
        let vol_sum = gpu.storage("vol_sum", n * layers as u64 * 4, usage);
        let vol_cnt = gpu.storage("vol_cnt", n * (layers / 4) as u64 * 4, usage);
        let stats = gpu.storage("stats", n * 16, usage);
        let g = gpu.storage("g", n * 4, usage);
        let depth = gpu.storage("d", n * 4, usage);
        let a = gpu.storage("a", n * 4, usage);
        let q = gpu.storage("q", n * 8, usage);
        let conf = gpu.storage("confidence", n * 8, usage);
        ActiveKeyframe {
            frame,
            t_wr,
            k: *k,
            reference_rgb: reference.rgb.clone(),
            xi_range,
            lambda,
            mp,
            n,
            update_bg: gpu.bind_group(&self.cost_update, &[&params_buf, &xf_buf, &rgb, &img_buf, &vol_sum, &vol_cnt]),
            minmax_bg: gpu.bind_group(&self.minmax, &[&params_buf, &vol_sum, &vol_cnt, &stats, &depth, &a]),
            weights_bg: gpu.bind_group(&self.weights, &[&params_buf, &rgb, &g]),
            dual_bg: gpu.bind_group(&self.dual, &[&params_buf, &g, &depth, &q]),
            primal_bg: gpu.bind_group(&self.primal, &[&params_buf, &g, &q, &a, &depth]),
            aux_bg: gpu.bind_group(&self.aux, &[&params_buf, &vol_sum, &vol_cnt, &stats, &depth, &a]),
            conf_bg: gpu.bind_group(&self.confidence, &[&params_buf, &vol_sum, &vol_cnt, &depth, &conf]),
            conf,
            params_buf,
            xf_buf,
            img_buf,
            _vol_sum: vol_sum,
            _vol_cnt: vol_cnt,
            stats,
            depth,
            rgb,
            q,
            frames_used: 0,
            frames_since_solve: 0,
            max_dist: 0.0,
        }
    }

    /// Accumulates one overlapping frame into the cost volume (eqs. 2-3).
    /// Returns false once the per-voxel counters are full (255 frames).
    pub fn add_frame(&self, gpu: &Gpu, akf: &mut ActiveKeyframe, img: &Frame, t_wm: Se3) -> bool {
        if akf.frames_used >= 250 {
            return false;
        }
        let kk = akf.k.k();
        let t_mr = t_wm.inverse().compose(&akf.t_wr);
        let m = kk * t_mr.r * akf.k.k_inv();
        let b = kk * t_mr.t;
        let xf = FrameXf {
            m: [0, 1, 2].map(|r| [m[(r, 0)] as f32, m[(r, 1)] as f32, m[(r, 2)] as f32, 0.0]),
            b: [b.x as f32, b.y as f32, b.z as f32, 0.0],
        };
        gpu.queue.write_buffer(&akf.xf_buf, 0, bytemuck::bytes_of(&xf));
        gpu.queue.write_buffer(&akf.img_buf, 0, bytemuck::cast_slice(&img.pack_rgba()));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.cost_update);
            pass.set_bind_group(0, &akf.update_bg, &[]);
            dispatch_2d(&mut pass, akf.mp.w, akf.mp.h);
        }
        gpu.queue.submit([enc.finish()]);
        akf.frames_used += 1;
        akf.frames_since_solve += 1;
        akf.max_dist = akf.max_dist.max((t_wm.t - akf.t_wr.t).norm());
        true
    }

    /// Solves the regularised inverse depth from the current cost volume
    /// (paper §2.2.3) and reads the result back.
    pub async fn solve(&self, gpu: &Gpu, params: &DtamParams, akf: &mut ActiveKeyframe) -> Keyframe {
        let (w, h, n) = (akf.mp.w, akf.mp.h, akf.n);
        let mut mp = akf.mp;
        gpu.queue.write_buffer(&akf.params_buf, 0, bytemuck::bytes_of(&mp));

        // Per-pixel cost statistics, initial d = a = arg min C, weights g.
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.minmax);
            pass.set_bind_group(0, &akf.minmax_bg, &[]);
            dispatch_2d(&mut pass, w, h);
            pass.set_pipeline(&self.weights);
            pass.set_bind_group(0, &akf.weights_bg, &[]);
            dispatch_2d(&mut pass, w, h);
        }
        let argmin_rb = gpu.readback("argmin_rb", n * 16);
        enc.copy_buffer_to_buffer(&akf.stats, 0, &argmin_rb, 0, n * 16);
        gpu.queue.submit([enc.finish()]);

        // Alternate primal-dual steps and the point-wise search while driving
        // theta to zero (paper §2.2.3 steps 1-3). The dual q starts at 0.
        let mut theta = params.theta_start;
        let mut iter = 0u32;
        let (mut sigma_q, mut sigma_d) = (params.sigma_q, params.sigma_d);
        let mut first = true;
        while theta > params.theta_end {
            mp.theta = theta;
            mp.sigma_q = sigma_q;
            mp.sigma_d = sigma_d;
            if params.accelerate {
                let w = 1.0 / (1.0 + 2.0 * sigma_d / theta).sqrt();
                sigma_d *= w;
                sigma_q /= w;
            }
            gpu.queue.write_buffer(&akf.params_buf, 0, bytemuck::bytes_of(&mp));
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            if first {
                // Reset the dual variable for this solve.
                enc.clear_buffer(&akf.q, 0, None);
                first = false;
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (pipe, bg) in [(&self.dual, &akf.dual_bg), (&self.primal, &akf.primal_bg), (&self.aux, &akf.aux_bg)] {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    dispatch_2d(&mut pass, w, h);
                }
            }
            gpu.queue.submit([enc.finish()]);
            let beta = if theta >= params.theta_switch { params.theta_beta_fast } else { params.theta_beta_slow };
            let factor = 1.0 - beta * iter as f32;
            iter += 1;
            if factor <= 0.0 || iter > 5000 {
                break;
            }
            theta *= factor;
        }

        let d_rb = gpu.readback("d_rb", n * 4);
        let conf_rb = gpu.readback("conf_rb", n * 8);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.confidence);
            pass.set_bind_group(0, &akf.conf_bg, &[]);
            dispatch_2d(&mut pass, w, h);
        }
        enc.copy_buffer_to_buffer(&akf.depth, 0, &d_rb, 0, n * 4);
        enc.copy_buffer_to_buffer(&akf.conf, 0, &conf_rb, 0, n * 8);
        gpu.queue.submit([enc.finish()]);
        let data = gpu.read_buffers(&[(&d_rb, n * 4), (&argmin_rb, n * 16), (&conf_rb, n * 8)]).await;
        let cf: Vec<[f32; 2]> = bytemuck::pod_collect_to_vec(&data[2]);
        let inv_depth: Vec<f32> = bytemuck::pod_collect_to_vec(&data[0]);
        let st: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&data[1]);
        let mut sorted: Vec<f32> = inv_depth.iter().copied().filter(|v| *v > 0.0).collect();
        sorted.sort_by(f32::total_cmp);
        let median_inv_depth =
            sorted.get(sorted.len() / 2).copied().unwrap_or((akf.xi_range.0 + akf.xi_range.1) / 2.0);
        akf.frames_since_solve = 0;
        Keyframe {
            id: 0,
            frame: akf.frame,
            pose: akf.t_wr,
            intrinsics: akf.k,
            rgb: akf.reference_rgb.clone(),
            inv_depth,
            argmin_inv_depth: st.iter().map(|s| s[2]).collect(),
            cost_range: st.iter().map(|s| s[1] - s[0]).collect(),
            confidence: cf.iter().map(|c| c[0]).collect(),
            photometric_error: cf.iter().map(|c| c[1]).collect(),
            xi_range: akf.xi_range,
            median_inv_depth,
            lambda: akf.lambda,
            frames_used: akf.frames_used,
            iterations: iter,
            baseline: (akf.max_dist * median_inv_depth as f64) as f32,
        }
    }
}
