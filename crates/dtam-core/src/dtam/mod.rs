//! DTAM: dense mapping (keyframe cost volumes + regularised inverse depth,
//! paper §2.2) and dense tracking (whole-image alignment against the model,
//! paper §2.3), on the GPU.

mod mapping;
mod tracking;

use std::sync::Arc;

pub use mapping::{ActiveKeyframe, Mapper};
pub use tracking::{DenseTracker, Prediction, TrackStats};

use crate::geom::{Intrinsics, Se3};
use crate::{Frame, Gpu};

#[derive(Clone, Debug)]
pub struct DtamParams {
    /// Inverse depth samples per pixel (S in the paper). Multiple of 4.
    pub layers: u32,
    /// g(u) = exp(-alpha |grad I|^beta), image intensities in [0, 1].
    pub alpha: f32,
    pub beta: f32,
    /// Huber epsilon.
    pub epsilon: f32,
    pub theta_start: f32,
    pub theta_end: f32,
    /// theta_{n+1} = theta_n (1 - beta_n n): beta_n = `theta_beta_fast` while
    /// theta >= `theta_switch`, else `theta_beta_slow` (paper §2.2.6).
    pub theta_beta_fast: f32,
    pub theta_beta_slow: f32,
    pub theta_switch: f32,
    /// Initial dual / primal steps (sigma_q sigma_d * 8 <= 1).
    pub sigma_q: f32,
    pub sigma_d: f32,
    /// Chambolle-Pock acceleration of the steps for the (1/theta)-strongly
    /// convex primal ([3] Alg. 2).
    pub accelerate: bool,
    /// Voxels observed by fewer frames are treated as unobserved.
    pub min_voxel_views: u32,
    /// Pyramid levels for tracking (level 0 = mapping resolution).
    pub track_levels: u32,
    /// Gauss-Newton iterations per level, coarsest first. 0 skips 6DOF on
    /// that level (the paper uses the coarse levels for rotation only).
    pub track_iterations: Vec<u32>,
    /// Photometric outlier threshold per level, coarsest first (paper §2.3.2).
    pub track_thresholds: Vec<f32>,
    /// Pyramid levels used for the rotation pre-alignment, coarsest first.
    pub rotation_levels: Vec<u32>,
    pub rotation_iterations: u32,
    /// Mesh vertices whose surface normal is more oblique than this to the
    /// keyframe's viewing ray are culled (degrees).
    pub max_oblique_deg: f32,
}

impl Default for DtamParams {
    fn default() -> Self {
        Self {
            layers: 64,
            alpha: 100.0,
            beta: 1.6,
            epsilon: 1e-4,
            theta_start: 0.2,
            theta_end: 1e-4,
            // The paper's example is 1e-3 / 1e-4 and notes smaller β with more
            // iterations gives higher quality; 1/4 of it (~470 iterations)
            // lets the regulariser fill textureless regions.
            theta_beta_fast: 2.5e-4,
            theta_beta_slow: 2.5e-5,
            theta_switch: 1e-3,
            sigma_q: 0.5,
            sigma_d: 0.25,
            accelerate: false,
            min_voxel_views: 3,
            track_levels: 4,
            track_iterations: vec![20, 20, 15, 10],
            track_thresholds: vec![0.25, 0.18, 0.12, 0.09],
            rotation_levels: vec![3, 2],
            rotation_iterations: 10,
            max_oblique_deg: 85.0,
        }
    }
}

/// A converged keyframe: reference image, pose and inverse depth map.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub id: usize,
    /// Video frame index of the reference image.
    pub frame: usize,
    /// `T_wr`.
    pub pose: Se3,
    pub intrinsics: Intrinsics,
    pub rgb: Vec<u8>,
    /// Regularised inverse depth ξ (paper eq. 6).
    pub inv_depth: Vec<f32>,
    /// Raw data-term minimum arg min_d C(u, d), for comparison (paper Fig. 3).
    pub argmin_inv_depth: Vec<f32>,
    /// C_max - C_min per pixel.
    pub cost_range: Vec<f32>,
    /// (C_mean - C(ξ)) / C_mean × fraction of depth samples observed: how
    /// uniquely the data term pins down the final depth (0 = textureless).
    pub confidence: Vec<f32>,
    /// Average photometric error C(u, ξ(u)) at the final depth.
    pub photometric_error: Vec<f32>,
    pub xi_range: (f32, f32),
    /// Median of the regularised inverse depth.
    pub median_inv_depth: f32,
    pub lambda: f32,
    pub frames_used: usize,
    pub iterations: u32,
    /// Largest camera distance from the reference among the frames used,
    /// divided by the median scene depth (parallax available to the data term).
    pub baseline: f32,
}

impl Keyframe {
    /// World-space colored points with confidence >= `min_confidence`.
    pub fn points(&self, step: usize, min_confidence: f32) -> Vec<([f32; 3], [u8; 3])> {
        let (w, h) = (self.intrinsics.width as usize, self.intrinsics.height as usize);
        let mut out = Vec::new();
        for y in (0..h).step_by(step.max(1)) {
            for x in (0..w).step_by(step.max(1)) {
                let i = y * w + x;
                let xi = self.inv_depth[i];
                if xi <= 0.0 || self.confidence[i] < min_confidence {
                    continue;
                }
                let ray = self.intrinsics.unproject([x as f64, y as f64]);
                let p = self.pose.transform(&(ray / xi as f64));
                out.push((
                    [p.x as f32, p.y as f32, p.z as f32],
                    [self.rgb[i * 3], self.rgb[i * 3 + 1], self.rgb[i * 3 + 2]],
                ));
            }
        }
        out
    }
}

/// The dense model plus the GPU machinery that builds and tracks against it.
pub struct Dtam {
    pub gpu: Gpu,
    pub intrinsics: Intrinsics,
    pub params: DtamParams,
    mapper: Mapper,
    tracker: DenseTracker,
    pub keyframes: Vec<Arc<Keyframe>>,
    /// The keyframe whose cost volume is still accumulating frames.
    pub active: Option<ActiveKeyframe>,
    /// Model index of the active keyframe once it has been solved at least once.
    active_id: Option<usize>,
}

impl Dtam {
    pub fn new(gpu: Gpu, intrinsics: Intrinsics, params: DtamParams) -> Self {
        let mapper = Mapper::new(&gpu);
        let tracker = DenseTracker::new(&gpu, intrinsics, &params);
        Self { gpu, intrinsics, params, mapper, tracker, keyframes: Vec::new(), active: None, active_id: None }
    }

    /// Starts a new keyframe (the previous active one stops accumulating).
    pub fn begin_keyframe(&mut self, frame_index: usize, reference: &Frame, t_wr: Se3, xi_range: (f32, f32), lambda: f32) {
        self.active = Some(self.mapper.begin(&self.gpu, &self.intrinsics, &self.params, frame_index, reference, t_wr, xi_range, lambda));
        self.active_id = None;
    }

    /// Adds an overlapping frame to the active keyframe's cost volume.
    pub fn add_to_keyframe(&mut self, frame: &Frame, t_wm: Se3) -> bool {
        match self.active.as_mut() {
            Some(akf) => self.mapper.add_frame(&self.gpu, akf, frame, t_wm),
            None => false,
        }
    }

    /// (Re-)solves the active keyframe and puts it into / refreshes it in the
    /// model used for tracking.
    pub async fn solve_keyframe(&mut self) -> Option<Arc<Keyframe>> {
        let akf = self.active.as_mut()?;
        let built = self.mapper.solve(&self.gpu, &self.params, akf).await;
        let id = self.active_id.unwrap_or(self.keyframes.len());
        let kf = Arc::new(Keyframe { id, ..built });
        match self.active_id {
            Some(id) => {
                self.keyframes[id] = kf.clone();
                self.tracker.update_model_keyframe(id, kf.clone());
            }
            None => {
                self.tracker.add_model_keyframe(&self.gpu, kf.clone(), akf.depth.clone(), akf.rgb.clone());
                self.keyframes.push(kf.clone());
                self.active_id = Some(id);
            }
        }
        Some(kf)
    }

    /// Builds a keyframe in one go from `frames` (image, `T_wm`) and adds it
    /// to the model.
    pub async fn add_keyframe(
        &mut self,
        frame_index: usize,
        reference: &Frame,
        t_wr: Se3,
        frames: &[(&Frame, Se3)],
        xi_range: (f32, f32),
        lambda: f32,
    ) -> Arc<Keyframe> {
        self.begin_keyframe(frame_index, reference, t_wr, xi_range, lambda);
        for (f, p) in frames {
            self.add_to_keyframe(f, *p);
        }
        self.solve_keyframe().await.unwrap()
    }

    /// Renders the model into a virtual camera at `t_wv`.
    pub async fn predict(&mut self, t_wv: Se3) -> Prediction {
        self.tracker.predict(&self.gpu, t_wv).await
    }

    /// Uploads a live frame and builds its pyramid in `slot` (0 or 1).
    pub fn set_live(&mut self, frame: &Frame, slot: usize) {
        self.tracker.set_live(&self.gpu, frame, slot);
    }

    /// Rotation `R_cur_prev` between the live frames in two slots.
    pub async fn rotation(&mut self, prev_slot: usize, cur_slot: usize) -> crate::geom::Mat3 {
        self.tracker.rotation(&self.gpu, &self.params, prev_slot, cur_slot).await
    }

    /// Photometric cost of the live frame in `slot` at pose `t_wl` against `prediction`.
    pub async fn cost(&mut self, slot: usize, prediction: &Prediction, t_wl: Se3) -> (f64, f64) {
        self.tracker.cost(&self.gpu, &self.params, slot, prediction, t_wl).await
    }

    /// Aligns the live frame in `slot` against the last prediction; returns `T_wl`.
    pub async fn align(&mut self, slot: usize, prediction: &Prediction, want_mask: bool) -> (Se3, TrackStats) {
        self.tracker.align(&self.gpu, &self.params, slot, prediction, want_mask).await
    }
}
