//! GPU sparse point tracker: Shi-Tomasi corner detection + pyramidal
//! Lucas-Kanade (KLT) with a forward-backward consistency check.
//!
//! Per frame, one GPU submission does: luma upload -> image pyramid -> KLT on
//! all live tracks -> corner candidates (best pixel per 16x16 tile). The CPU
//! then culls bad tracks and spawns new ones in empty regions.

use bytemuck::{Pod, Zeroable};

use crate::gpu::{Gpu, dispatch_2d};

const MAX_LEVELS: usize = 6;
const TILE: u32 = 16;

#[derive(Clone, Debug)]
pub struct TrackerParams {
    /// Pyramid levels used by KLT (clamped to what the image size allows, max 6).
    pub pyramid_levels: u32,
    /// KLT window half-size in pixels at every pyramid level.
    pub window_radius: u32,
    pub max_iterations: u32,
    /// Cap on simultaneously tracked points.
    pub max_features: usize,
    /// New points are not spawned closer than this to an existing one (px).
    pub min_distance: f32,
    /// Candidate must reach this fraction of the frame's best corner score.
    pub quality_level: f32,
    /// Absolute minimum corner score (min eigenvalue, intensities in [0,1]).
    pub min_response: f32,
    /// Max forward-backward round-trip error (px) before a track is dropped.
    pub max_fb_error: f32,
    /// Max mean absolute intensity difference over the window (0..1).
    pub max_residual: f32,
    /// Ignore corners within this many pixels of the image edge.
    pub border: u32,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            pyramid_levels: 4,
            window_radius: 7,
            max_iterations: 20,
            max_features: 800,
            min_distance: 12.0,
            quality_level: 0.02,
            min_response: 1e-5,
            max_fb_error: 1.0,
            max_residual: 0.08,
            border: 12,
        }
    }
}

/// A tracked point as seen in one frame.
#[derive(Clone, Copy, Debug)]
pub struct TrackedPoint {
    /// Stable id across frames.
    pub id: u32,
    /// Pixel position (x right, y down), pixel centers at integers.
    pub pos: [f32; 2],
    /// Frames since this track was spawned (0 = new this frame).
    pub age: u32,
    /// KLT photometric residual this frame (0 for new points).
    pub residual: f32,
    /// Forward-backward error in px this frame (0 for new points).
    pub fb_error: f32,
}

#[derive(Clone, Debug, Default)]
pub struct FrameTracks {
    pub points: Vec<TrackedPoint>,
    /// Tracks spawned this frame.
    pub born: u32,
    /// Tracks from the previous frame that were dropped this frame.
    pub lost: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UnpackParams {
    dst_off: u32,
    w: u32,
    h: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleParams {
    src_off: u32,
    src_w: u32,
    src_h: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CornerParams {
    off: u32,
    w: u32,
    h: u32,
    border: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct KltParams {
    levels: [[u32; 4]; MAX_LEVELS],
    slot_size: u32,
    num_levels: u32,
    prev_slot: u32,
    cur_slot: u32,
    num_points: u32,
    radius: i32,
    max_iters: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointIn {
    pos: [f32; 2],
    guess: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointOut {
    pos: [f32; 2],
    back: [f32; 2],
    residual: f32,
    status: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Candidate {
    xy: u32,
    score: f32,
}

#[derive(Clone, Copy)]
struct Level {
    off: u32,
    w: u32,
    h: u32,
}

struct Track {
    id: u32,
    pos: [f32; 2],
    age: u32,
}

/// Per-slot (prev/cur frame alternate between two pyramid slots) bind groups.
struct SlotBindings {
    unpack: wgpu::BindGroup,
    downsample: Vec<wgpu::BindGroup>,
    corners: wgpu::BindGroup,
}

pub struct KltTracker {
    gpu: Gpu,
    params: TrackerParams,
    width: u32,
    height: u32,
    levels: Vec<Level>,
    slot_size: u32,
    tiles: (u32, u32),

    unpack_pipe: wgpu::ComputePipeline,
    downsample_pipe: wgpu::ComputePipeline,
    corners_pipe: wgpu::ComputePipeline,
    klt_pipe: wgpu::ComputePipeline,

    luma_buf: wgpu::Buffer,
    klt_params_buf: wgpu::Buffer,
    pts_in_buf: wgpu::Buffer,
    pts_out_buf: wgpu::Buffer,
    cands_buf: wgpu::Buffer,
    pts_readback: wgpu::Buffer,
    cands_readback: wgpu::Buffer,

    slots: [SlotBindings; 2],
    klt_bind: wgpu::BindGroup,

    frame_index: u64,
    tracks: Vec<Track>,
    next_id: u32,
}

impl KltTracker {
    pub fn new(gpu: Gpu, width: u32, height: u32, params: TrackerParams) -> Self {
        // Pyramid: stop before the top level gets smaller than a few KLT windows.
        let min_dim = 4 * (2 * params.window_radius + 1);
        let mut levels = vec![Level { off: 0, w: width, h: height }];
        while levels.len() < (params.pyramid_levels as usize).clamp(1, MAX_LEVELS) {
            let last = *levels.last().unwrap();
            let (w, h) = (last.w.div_ceil(2), last.h.div_ceil(2));
            if w.min(h) < min_dim {
                break;
            }
            levels.push(Level { off: last.off + last.w * last.h, w, h });
        }
        let last = levels.last().unwrap();
        let slot_size = last.off + last.w * last.h;
        let tiles = (width.div_ceil(TILE), height.div_ceil(TILE));
        let cap = params.max_features as u64;

        let unpack_pipe = gpu.compute_pipeline("unpack", include_str!("shaders/unpack.wgsl"));
        let downsample_pipe =
            gpu.compute_pipeline("downsample", include_str!("shaders/downsample.wgsl"));
        let corners_pipe = gpu.compute_pipeline("corners", include_str!("shaders/corners.wgsl"));
        let klt_pipe = gpu.compute_pipeline("klt", include_str!("shaders/klt.wgsl"));

        let copy_dst = wgpu::BufferUsages::COPY_DST;
        let copy_src = wgpu::BufferUsages::COPY_SRC;
        let luma_buf = gpu.storage("luma", (width * height).div_ceil(4) as u64 * 4, copy_dst);
        let pyr_buf = gpu.storage("pyramid", 2 * slot_size as u64 * 4, wgpu::BufferUsages::empty());
        let pts_in_buf = gpu.storage("pts_in", cap * size_of::<PointIn>() as u64, copy_dst);
        let pts_out_size = cap * size_of::<PointOut>() as u64;
        let pts_out_buf = gpu.storage("pts_out", pts_out_size, copy_src);
        let cands_size = (tiles.0 * tiles.1) as u64 * size_of::<Candidate>() as u64;
        let cands_buf = gpu.storage("candidates", cands_size, copy_src);
        let pts_readback = gpu.readback("pts_readback", pts_out_size);
        let cands_readback = gpu.readback("cands_readback", cands_size);
        let klt_params_buf = gpu.uniform("klt_params", &KltParams::zeroed());

        let slots = [0u32, 1].map(|slot| {
            let base = slot * slot_size;
            let l0 = levels[0];
            let unpack_u = gpu.uniform(
                "unpack_params",
                &UnpackParams { dst_off: base, w: l0.w, h: l0.h, _pad: 0 },
            );
            let downsample = levels
                .windows(2)
                .map(|pair| {
                    let (s, d) = (pair[0], pair[1]);
                    let u = gpu.uniform(
                        "downsample_params",
                        &DownsampleParams {
                            src_off: base + s.off,
                            src_w: s.w,
                            src_h: s.h,
                            dst_off: base + d.off,
                            dst_w: d.w,
                            dst_h: d.h,
                            _pad: [0; 2],
                        },
                    );
                    gpu.bind_group(&downsample_pipe, &[&u, &pyr_buf])
                })
                .collect();
            let corners_u = gpu.uniform(
                "corner_params",
                &CornerParams { off: base, w: l0.w, h: l0.h, border: params.border },
            );
            SlotBindings {
                unpack: gpu.bind_group(&unpack_pipe, &[&unpack_u, &luma_buf, &pyr_buf]),
                downsample,
                corners: gpu.bind_group(&corners_pipe, &[&corners_u, &pyr_buf, &cands_buf]),
            }
        });
        let klt_bind = gpu.bind_group(
            &klt_pipe,
            &[&klt_params_buf, &pyr_buf, &pts_in_buf, &pts_out_buf],
        );

        Self {
            gpu,
            params,
            width,
            height,
            levels,
            slot_size,
            tiles,
            unpack_pipe,
            downsample_pipe,
            corners_pipe,
            klt_pipe,
            luma_buf,
            klt_params_buf,
            pts_in_buf,
            pts_out_buf,
            cands_buf,
            pts_readback,
            cands_readback,
            slots,
            klt_bind,
            frame_index: 0,
            tracks: Vec::new(),
            next_id: 0,
        }
    }

    pub fn params(&self) -> &TrackerParams {
        &self.params
    }

    pub fn num_pyramid_levels(&self) -> usize {
        self.levels.len()
    }

    /// Tracks existing points into this frame and spawns new ones.
    /// `luma` is `width * height` 8-bit intensities.
    pub async fn process(&mut self, luma: &[u8]) -> FrameTracks {
        assert_eq!(luma.len(), (self.width * self.height) as usize, "luma size mismatch");
        let gpu = self.gpu.clone();
        let cur = (self.frame_index % 2) as usize;
        let prev = 1 - cur;

        let mut padded = luma.to_vec();
        padded.resize(luma.len().next_multiple_of(4), 0);
        gpu.queue.write_buffer(&self.luma_buf, 0, &padded);

        let num_points = if self.frame_index > 0 { self.tracks.len() } else { 0 };
        if num_points > 0 {
            let pts: Vec<PointIn> =
                self.tracks.iter().map(|t| PointIn { pos: t.pos, guess: t.pos }).collect();
            gpu.queue.write_buffer(&self.pts_in_buf, 0, bytemuck::cast_slice(&pts));
            let mut levels = [[0u32; 4]; MAX_LEVELS];
            for (dst, l) in levels.iter_mut().zip(&self.levels) {
                *dst = [l.off, l.w, l.h, 0];
            }
            let kp = KltParams {
                levels,
                slot_size: self.slot_size,
                num_levels: self.levels.len() as u32,
                prev_slot: prev as u32,
                cur_slot: cur as u32,
                num_points: num_points as u32,
                radius: self.params.window_radius as i32,
                max_iters: self.params.max_iterations,
                _pad: 0,
            };
            gpu.queue.write_buffer(&self.klt_params_buf, 0, bytemuck::bytes_of(&kp));
        }

        let pts_bytes = (num_points * size_of::<PointOut>()) as u64;
        let cands_bytes = (self.tiles.0 * self.tiles.1) as u64 * size_of::<Candidate>() as u64;
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracker"),
                timestamp_writes: None,
            });
            let slot = &self.slots[cur];
            pass.set_pipeline(&self.unpack_pipe);
            pass.set_bind_group(0, &slot.unpack, &[]);
            dispatch_2d(&mut pass, self.width, self.height);

            pass.set_pipeline(&self.downsample_pipe);
            for (bg, lvl) in slot.downsample.iter().zip(&self.levels[1..]) {
                pass.set_bind_group(0, bg, &[]);
                dispatch_2d(&mut pass, lvl.w, lvl.h);
            }

            if num_points > 0 {
                pass.set_pipeline(&self.klt_pipe);
                pass.set_bind_group(0, &self.klt_bind, &[]);
                pass.dispatch_workgroups((num_points as u32).div_ceil(64), 1, 1);
            }

            pass.set_pipeline(&self.corners_pipe);
            pass.set_bind_group(0, &slot.corners, &[]);
            pass.dispatch_workgroups(self.tiles.0, self.tiles.1, 1);
        }
        if num_points > 0 {
            enc.copy_buffer_to_buffer(&self.pts_out_buf, 0, &self.pts_readback, 0, pts_bytes);
        }
        enc.copy_buffer_to_buffer(&self.cands_buf, 0, &self.cands_readback, 0, cands_bytes);
        gpu.queue.submit([enc.finish()]);

        let mut reads = vec![(&self.cands_readback, cands_bytes)];
        if num_points > 0 {
            reads.push((&self.pts_readback, pts_bytes));
        }
        let data = gpu.read_buffers(&reads).await;
        let cands: Vec<Candidate> = bytemuck::pod_collect_to_vec(&data[0]);
        let outs: Vec<PointOut> =
            if num_points > 0 { bytemuck::pod_collect_to_vec(&data[1]) } else { Vec::new() };

        let result = self.update_tracks(&outs, &cands);
        self.frame_index += 1;
        result
    }

    fn update_tracks(&mut self, outs: &[PointOut], cands: &[Candidate]) -> FrameTracks {
        let p = &self.params;
        let prev_count = self.tracks.len();

        // Occupancy grid with cell size = min_distance; a candidate is rejected
        // if any point lies in its 3x3 cell neighbourhood.
        let cell = p.min_distance.max(1.0);
        let gw = (self.width as f32 / cell).ceil() as usize + 1;
        let gh = (self.height as f32 / cell).ceil() as usize + 1;
        let mut occupied = vec![false; gw * gh];
        let cell_of = |pos: [f32; 2]| {
            let cx = ((pos[0] / cell) as usize).min(gw - 1);
            let cy = ((pos[1] / cell) as usize).min(gh - 1);
            (cx, cy)
        };

        let mut points = Vec::with_capacity(p.max_features);
        let mut survivors = Vec::with_capacity(prev_count);
        // Tracks are kept oldest-first, so when two collapse into one cell
        // (e.g. drifting along an edge) the younger one is dropped.
        for (t, o) in self.tracks.drain(..).zip(outs) {
            let fb = ((o.back[0] - t.pos[0]).powi(2) + (o.back[1] - t.pos[1]).powi(2)).sqrt();
            if o.status != 1 || fb > p.max_fb_error || o.residual > p.max_residual {
                continue;
            }
            let (cx, cy) = cell_of(o.pos);
            if std::mem::replace(&mut occupied[cy * gw + cx], true) {
                continue;
            }
            let track = Track { id: t.id, pos: o.pos, age: t.age + 1 };
            points.push(TrackedPoint {
                id: track.id,
                pos: track.pos,
                age: track.age,
                residual: o.residual,
                fb_error: fb,
            });
            survivors.push(track);
        }
        let lost = (prev_count - survivors.len()) as u32;

        let best = cands.iter().map(|c| c.score).fold(0.0f32, f32::max);
        let threshold = (best * p.quality_level).max(p.min_response);
        let mut sorted: Vec<_> = cands.iter().filter(|c| c.score >= threshold).collect();
        sorted.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut born = 0;
        for c in sorted {
            if survivors.len() >= p.max_features {
                break;
            }
            let pos = [(c.xy & 0xffff) as f32, (c.xy >> 16) as f32];
            let (cx, cy) = cell_of(pos);
            let free = (cy.saturating_sub(1)..=(cy + 1).min(gh - 1)).all(|y| {
                (cx.saturating_sub(1)..=(cx + 1).min(gw - 1)).all(|x| !occupied[y * gw + x])
            });
            if !free {
                continue;
            }
            occupied[cy * gw + cx] = true;
            let id = self.next_id;
            self.next_id += 1;
            survivors.push(Track { id, pos, age: 0 });
            points.push(TrackedPoint { id, pos, age: 0, residual: 0.0, fb_error: 0.0 });
            born += 1;
        }

        self.tracks = survivors;
        FrameTracks { points, born, lost }
    }
}
