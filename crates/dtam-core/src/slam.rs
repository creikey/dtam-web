//! Streaming SLAM driver, one frame at a time (video files and live cameras):
//! KLT tracks -> self-calibration -> feature bootstrap (retried until the
//! camera has moved enough) -> first DTAM keyframe -> dense tracking with
//! coverage-triggered keyframes, and lost-tracking detection.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::calib::{FocalParams, estimate_focal};
use crate::dtam::{Dtam, DtamParams, Keyframe, TrackStats};
use crate::geom::{Intrinsics, Se3};
use crate::sfm::{BootstrapParams, bootstrap};
use crate::{Frame, FrameTracks, Gpu};

#[derive(Clone, Debug)]
pub struct SlamParams {
    pub focal: FocalParams,
    pub bootstrap: BootstrapParams,
    pub dtam: DtamParams,
    /// Already-tracked frames before a new keyframe that seed its cost
    /// volume (every later tracked frame is added as it arrives).
    pub keyframe_frames_before: usize,
    /// Only used with `first_keyframe`: frames after it in its window.
    pub keyframe_frames_after: usize,
    /// Re-solve the active keyframe after this many new frames.
    pub resolve_every: usize,
    /// A new keyframe joins the tracking model once it holds this many frames
    /// (re-solved every `resolve_every / 2` frames until then).
    pub publish_after_frames: usize,
    /// Past frames seed a new keyframe only within this baseline (× depth)
    /// and rotation of it.
    pub seed_max_baseline: f64,
    pub seed_max_angle_deg: f64,
    /// A frame joins the active keyframe's cost volume only within this
    /// baseline (× the keyframe's median depth) and rotation of it,
    pub max_keyframe_baseline: f64,
    pub max_keyframe_angle_deg: f64,
    /// and if tracking used at least this fraction of predicted pixels.
    pub min_used_for_mapping: f32,
    /// Start a new keyframe when the predicted model covers less than this.
    pub new_keyframe_coverage: f32,
    pub min_keyframe_spacing: usize,
    /// Depth range margins around the predicted/sparse inverse depths.
    pub xi_margin: (f32, f32),
    /// Frames of KLT tracks collected before trying the feature bootstrap.
    pub bootstrap_window: usize,
    /// If the bootstrap fails, drop this many frames from the window and retry.
    pub bootstrap_retry_step: usize,
    /// Horizontal FOV assumed when self-calibration fails (degrees).
    pub fallback_hfov_deg: f64,
    /// Longest side of the DTAM mapping/tracking resolution (the input is
    /// halved until it fits).
    pub mapping_max_dim: u32,
    /// Stop creating keyframes past this many (bounds memory).
    pub max_keyframes: usize,
    /// Consecutive failed frames before declaring tracking lost.
    pub lost_after: usize,
    /// Use this focal length (px, video resolution) instead of self-calibration
    /// and keep it fixed in the bootstrap bundle adjustment.
    pub fixed_focal: Option<f64>,
    /// Debug: reference `T_wc` poses for frames `start..` in their own frame
    /// (identity at `start`, median depth 1); each frame's photometric cost at
    /// the reference vs the tracked pose is logged.
    pub debug_reference: Option<(usize, Vec<Se3>)>,
    /// Largest ratio xi_max / xi_min of a keyframe's search range.
    pub max_depth_ratio: f32,
}

impl Default for SlamParams {
    fn default() -> Self {
        Self {
            focal: FocalParams::default(),
            bootstrap: BootstrapParams::default(),
            dtam: DtamParams::default(),
            keyframe_frames_before: 30,
            keyframe_frames_after: 6,
            resolve_every: 20,
            publish_after_frames: 0,
            seed_max_baseline: 0.3,
            seed_max_angle_deg: 15.0,
            max_keyframe_baseline: f64::INFINITY,
            max_keyframe_angle_deg: f64::INFINITY,
            min_used_for_mapping: 0.8,
            new_keyframe_coverage: 0.92,
            min_keyframe_spacing: 8,
            xi_margin: (0.5, 1.6),
            max_depth_ratio: 36.0,
            bootstrap_window: 60,
            bootstrap_retry_step: 15,
            fallback_hfov_deg: 60.0,
            mapping_max_dim: 512,
            max_keyframes: 64,
            lost_after: 15,
            fixed_focal: None,
            debug_reference: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseSource {
    /// Feature-based bootstrap (bundle adjusted).
    Bootstrap,
    /// Dense whole-image alignment against the model.
    Dense,
    /// Tracking failed; pose is the motion-model prediction.
    Predicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Collecting KLT tracks for the feature bootstrap.
    Bootstrapping { collected: usize, needed: usize },
    /// Dense tracking and mapping.
    Dense,
    /// Dense tracking failed repeatedly; still trying from the last good pose.
    Lost,
}

#[derive(Clone, Debug)]
pub enum SlamEvent {
    Stage(String),
    Phase(Phase),
    /// KLT feature tracks of a frame (only while bootstrapping).
    Klt { frame: usize, tracks: crate::FrameTracks },
    /// Intrinsics at video resolution: self-calibrated guess, bundle-adjusted.
    Intrinsics { self_calibrated: f64, refined: Intrinsics, mapping: Intrinsics },
    /// `T_wc` of a frame. World = first frame's camera, median bootstrap depth = 1.
    Pose { frame: usize, pose: Se3, source: PoseSource },
    Tracking { frame: usize, stats: TrackStats },
    /// A new keyframe, or an update of an existing one (same `id`) after its
    /// cost volume received more frames.
    Keyframe(Arc<Keyframe>),
}

fn tracking_ok(s: &TrackStats) -> bool {
    s.used_fraction > 0.5 && s.rmse.is_finite() && s.rmse < 0.08
}

/// The whole pipeline as a state machine fed one frame at a time.
pub struct Slam {
    gpu: Gpu,
    pub params: SlamParams,
    input_size: (u32, u32),
    map_downsample: u32,
    klt: crate::SlamPipeline,
    phase: Phase,
    frame_index: usize,
    /// First frame of the bootstrap window and its KLT tracks.
    window_start: usize,
    window: Vec<FrameTracks>,
    /// Recent frames at mapping resolution.
    small: BTreeMap<usize, Frame>,
    dtam: Option<Dtam>,
    pub intrinsics: Option<Intrinsics>,
    poses: Vec<Option<Se3>>,
    tracked: Vec<bool>,
    last_good: Se3,
    last_kf_frame: usize,
    last_range: (f32, f32),
    failures: usize,
    /// Seed frames of the active keyframe (not counted towards publishing).
    kf_seeded: usize,
    ref_anchor: Option<(Se3, f64)>,
}

impl Slam {
    pub fn new(gpu: Gpu, input_size: (u32, u32), params: SlamParams) -> Self {
        let mut map_downsample = 0;
        while (input_size.0.max(input_size.1) >> map_downsample) > params.mapping_max_dim {
            map_downsample += 1;
        }
        let klt = crate::SlamPipeline::new(gpu.clone(), input_size.0, input_size.1, crate::TrackerParams::default());
        let needed = params.bootstrap_window;
        Self {
            gpu,
            params,
            input_size,
            map_downsample,
            klt,
            phase: Phase::Bootstrapping { collected: 0, needed },
            frame_index: 0,
            window_start: 0,
            window: Vec::new(),
            small: BTreeMap::new(),
            dtam: None,
            intrinsics: None,
            poses: Vec::new(),
            tracked: Vec::new(),
            last_good: Se3::identity(),
            last_kf_frame: 0,
            last_range: (0.1, 10.0),
            failures: 0,
            kf_seeded: 0,
            ref_anchor: None,
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn keyframes(&self) -> &[Arc<Keyframe>] {
        self.dtam.as_ref().map_or(&[], |d| &d.keyframes)
    }

    /// Mapping-resolution intrinsics (once bootstrapped).
    pub fn mapping_intrinsics(&self) -> Option<Intrinsics> {
        self.dtam.as_ref().map(|d| d.intrinsics)
    }

    /// Frame index the next `push` will get.
    pub fn next_frame_index(&self) -> usize {
        self.frame_index
    }

    fn downsample(&self, frame: &Frame) -> Frame {
        let mut f = frame.clone();
        for _ in 0..self.map_downsample {
            f = f.downsample2();
        }
        f
    }

    /// Processes the next frame (at `input_size`), emitting events.
    pub async fn push(&mut self, frame: &Frame, emit: &mut impl FnMut(SlamEvent)) {
        assert_eq!((frame.width, frame.height), self.input_size, "frame size changed");
        let f = self.frame_index;
        self.frame_index += 1;
        self.poses.push(None);
        self.tracked.push(false);
        let small = self.downsample(frame);
        self.small.insert(f, small.clone());

        match self.phase {
            Phase::Bootstrapping { needed, .. } => {
                let tracks = self.klt.process(frame).await.tracks;
                emit(SlamEvent::Klt { frame: f, tracks: tracks.clone() });
                self.window.push(tracks);
                self.phase = Phase::Bootstrapping { collected: self.window.len(), needed };
                emit(SlamEvent::Phase(self.phase));
                if self.window.len() >= needed {
                    self.try_bootstrap(emit).await;
                }
            }
            Phase::Dense | Phase::Lost => self.track(f, &small, emit).await,
        }

        // Bound memory: keep only frames still needed for keyframe seeding
        // (and the bootstrap window while bootstrapping).
        let keep_from = match self.phase {
            Phase::Bootstrapping { .. } => self.window_start,
            _ => f.saturating_sub(self.params.keyframe_frames_before + 2),
        };
        self.small.retain(|&i, _| i >= keep_from);
    }

    /// Starts over (e.g. after tracking was lost).
    pub fn reset(&mut self) {
        let fresh = Slam::new(self.gpu.clone(), self.input_size, self.params.clone());
        let frame_index = self.frame_index;
        *self = fresh;
        self.frame_index = frame_index;
        self.window_start = frame_index;
        self.poses = vec![None; frame_index];
        self.tracked = vec![false; frame_index];
    }

    async fn try_bootstrap(&mut self, emit: &mut impl FnMut(SlamEvent)) {
        let (w, h) = self.input_size;
        let params = self.params.clone();
        emit(SlamEvent::Stage("self-calibrating focal length".into()));
        let f0 = match params.fixed_focal.map(Ok).unwrap_or_else(|| estimate_focal(&self.window, w, h, &params.focal).map(|e| e.focal_px)) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("focal self-calibration failed ({e}); assuming {}° FOV", params.fallback_hfov_deg);
                w as f64 / 2.0 / (params.fallback_hfov_deg / 2.0).to_radians().tan()
            }
        };
        emit(SlamEvent::Stage("bootstrapping with features".into()));
        let bp = BootstrapParams {
            refine_focal: params.bootstrap.refine_focal && params.fixed_focal.is_none(),
            max_frames: self.window.len(),
            ..params.bootstrap.clone()
        };
        let boot = match bootstrap(&self.window, Intrinsics::centered(f0, w, h), &bp) {
            Ok(b) => b,
            Err(e) => {
                emit(SlamEvent::Stage(format!("bootstrap failed ({e}); keep moving the camera sideways")));
                let drop = params.bootstrap_retry_step.min(self.window.len());
                self.window.drain(..drop);
                self.window_start += drop;
                return;
            }
        };
        let base = self.window_start;
        let n0 = boot.poses.len();
        let first_small = self.small.get(&base).cloned().expect("bootstrap window frame");
        let k_map = boot.intrinsics.scaled(first_small.width as f64 / w as f64);
        self.intrinsics = Some(boot.intrinsics);
        emit(SlamEvent::Intrinsics { self_calibrated: f0, refined: boot.intrinsics, mapping: k_map });
        for (i, p) in boot.poses.iter().enumerate() {
            self.poses[base + i] = Some(*p);
            self.tracked[base + i] = true;
            emit(SlamEvent::Pose { frame: base + i, pose: *p, source: PoseSource::Bootstrap });
        }

        let mut dtam = Dtam::new(self.gpu.clone(), k_map, params.dtam.clone());
        let r0 = n0 / 2;
        emit(SlamEvent::Stage(format!("mapping keyframe 0 (frame {})", base + r0)));
        let t_rw = boot.poses[r0].inverse();
        let mut xis: Vec<f32> =
            boot.points.iter().map(|p| t_rw.transform(&p.pos).z).filter(|z| *z > 1e-3).map(|z| 1.0 / z as f32).collect();
        xis.sort_by(f32::total_cmp);
        if xis.is_empty() {
            emit(SlamEvent::Stage("bootstrap produced no points; retrying".into()));
            self.window.drain(..params.bootstrap_retry_step.min(self.window.len()));
            self.window_start += params.bootstrap_retry_step;
            return;
        }
        let pct = |v: &[f32], p: f32| v[((v.len() - 1) as f32 * p) as usize];
        let range0 = (pct(&xis, 0.02) * params.xi_margin.0, pct(&xis, 0.98) * params.xi_margin.1);
        let reference = self.small[&(base + r0)].clone();
        dtam.begin_keyframe(base + r0, &reference, boot.poses[r0], range0, 1.0);
        for i in 0..n0 {
            if i != r0
                && let Some(fr) = self.small.get(&(base + i))
            {
                dtam.add_to_keyframe(fr, boot.poses[i]);
            }
        }
        if let Some(kf) = dtam.solve_keyframe(0).await {
            emit(SlamEvent::Keyframe(kf));
        }
        // Model predictions for the bootstrap frames too (display / occlusion).
        for (i, p) in boot.poses.iter().enumerate() {
            let pred = dtam.predict(*p).await;
            emit(SlamEvent::Tracking { frame: base + i, stats: pred.to_stats() });
        }
        let last = base + n0 - 1;
        dtam.set_live(&self.small[&last], last % 2);
        self.last_good = boot.poses[n0 - 1];
        self.last_kf_frame = base + r0;
        self.last_range = range0;
        self.dtam = Some(dtam);
        self.window.clear();
        self.phase = Phase::Dense;
        emit(SlamEvent::Phase(self.phase));
        emit(SlamEvent::Stage("dense tracking".into()));
    }

    async fn track(&mut self, f: usize, live: &Frame, emit: &mut impl FnMut(SlamEvent)) {
        let params = self.params.clone();
        let dtam = self.dtam.as_mut().expect("dense phase has a model");
        let slot = f % 2;
        dtam.set_live(live, slot);
        let prev = self.poses[f - 1].unwrap_or(self.last_good);
        // Rotation pre-alignment against the previous frame (in the other slot).
        let r_cur_prev = dtam.rotation(1 - slot, slot).await;
        let prev2 = if f >= 2 { self.poses[f - 2].unwrap_or(prev) } else { prev };
        let velocity = prev2.inverse().compose(&prev);
        let predicted = prev.compose(&Se3::new(r_cur_prev.transpose(), velocity.t));

        // Track from the motion-model prediction; if that fails, retry once
        // from the last good pose (no global relocalisation, as in the
        // paper's evaluation).
        let mut candidates = vec![predicted];
        let mut prediction = None;
        let mut result: Option<(Se3, TrackStats)> = None;
        let mut attempt = 0;
        while attempt < candidates.len() {
            let pred = dtam.predict(candidates[attempt]).await;
            if pred.coverage >= 0.02 {
                let (pose, stats) = dtam.align(slot, &pred, true).await;
                if tracking_ok(&stats) {
                    prediction = Some(pred);
                    result = Some((pose, stats));
                    break;
                }
            }
            if attempt == 0 {
                candidates.push(self.last_good);
            }
            prediction.get_or_insert(pred);
            attempt += 1;
        }
        let prediction = prediction.unwrap();
        let (pose, source, stats) = match result {
            Some((pose, stats)) => (pose, PoseSource::Dense, stats),
            None => (predicted, PoseSource::Predicted, prediction.to_stats()),
        };

        if let Some((a, refs)) = &params.debug_reference
            && f >= *a
            && f - a < refs.len()
        {
            if f == *a {
                let p50 = prediction.inv_depth_percentiles[1].max(1e-3) as f64;
                self.ref_anchor = Some((pose, 1.0 / p50));
            }
            if let Some((anchor, scale)) = self.ref_anchor {
                let seg = refs[f - a];
                let reference = anchor.compose(&Se3::new(seg.r, seg.t * scale));
                let (c_ref, u_ref) = dtam.cost(slot, &prediction, reference).await;
                let (c_trk, u_trk) = dtam.cost(slot, &prediction, pose).await;
                let d = reference.inverse().compose(&pose);
                eprintln!(
                    "ref {f}: cost ref {c_ref:.5} (used {u_ref:.2}) tracked {c_trk:.5} (used {u_trk:.2}) | tracked vs ref: rot {:.2}° trans {:.4}",
                    d.rotation_angle().to_degrees(),
                    d.t.norm()
                );
            }
        }

        self.poses[f] = Some(pose);
        self.tracked[f] = source == PoseSource::Dense;
        if source == PoseSource::Dense {
            self.last_good = pose;
            self.failures = 0;
            if self.phase == Phase::Lost {
                self.phase = Phase::Dense;
                emit(SlamEvent::Phase(self.phase));
                emit(SlamEvent::Stage("tracking recovered".into()));
            }
        } else {
            self.failures += 1;
            if self.failures >= params.lost_after && self.phase != Phase::Lost {
                self.phase = Phase::Lost;
                emit(SlamEvent::Phase(self.phase));
                emit(SlamEvent::Stage("tracking lost — return to a mapped view or reset".into()));
            }
        }
        let stats_used = stats.used_fraction;
        emit(SlamEvent::Pose { frame: f, pose, source });
        emit(SlamEvent::Tracking { frame: f, stats });

        // Mapping (paper §2.2.1): the active keyframe keeps averaging in every
        // well-tracked frame, and its regularised solution is refreshed.
        let nearby = dtam.keyframes.last().zip(dtam.active.as_ref()).is_some_and(|(kf, akf)| {
            let rel = akf.t_wr.inverse().compose(&pose);
            rel.t.norm() * kf.median_inv_depth as f64 <= params.max_keyframe_baseline
                && rel.rotation_angle().to_degrees() <= params.max_keyframe_angle_deg
        });
        if source == PoseSource::Dense && nearby && stats_used >= params.min_used_for_mapping {
            dtam.add_to_keyframe(live, pose);
            let every = if dtam.active_published() { params.resolve_every } else { (params.resolve_every / 2).max(1) };
            if dtam.active.as_ref().is_some_and(|a| a.frames_since_solve >= every)
                && let Some(kf) = dtam.solve_keyframe(self.kf_seeded + params.publish_after_frames).await
            {
                emit(SlamEvent::Keyframe(kf));
            }
        }

        // Keyframe management (paper §2.4): new keyframe when the model no
        // longer covers enough of the view.
        if source == PoseSource::Dense
            && f - self.last_kf_frame >= params.min_keyframe_spacing
            && prediction.coverage < params.new_keyframe_coverage
            && dtam.keyframes.len() < params.max_keyframes
            && dtam.active_published()
        {
            let [p2, p50, p98] = prediction.inv_depth_percentiles;
            // The paper does not say how D = [ξmin, ξmax] is chosen; we span the
            // predicted inverse depths (with margins), capped around the median.
            let range = if p2 > 0.0 && p98 > p2 {
                let spread = params.max_depth_ratio.sqrt();
                ((p2 * params.xi_margin.0).max(p50 / spread), (p98 * params.xi_margin.1).min(p50 * spread))
            } else {
                self.last_range
            };
            // λ = 1 / (1 + 0.5 d_min), d_min = nearest predicted depth (§2.2.6).
            let min_depth = params.xi_margin.1 / range.1;
            let lambda = 1.0 / (1.0 + 0.5 * min_depth);
            if dtam.active.as_ref().is_some_and(|a| a.frames_since_solve > 0)
                && let Some(kf) = dtam.solve_keyframe(0).await
            {
                emit(SlamEvent::Keyframe(kf));
            }
            emit(SlamEvent::Stage(format!("mapping keyframe {} (frame {f})", dtam.keyframes.len())));
            dtam.begin_keyframe(f, live, pose, range, lambda);
            let mut seeded = 0;
            let depth = 2.0 / (range.0 / params.xi_margin.0 + range.1 / params.xi_margin.1) as f64;
            for i in f.saturating_sub(params.keyframe_frames_before)..f {
                if self.tracked[i]
                    && let (Some(p), Some(fr)) = (self.poses[i], self.small.get(&i))
                    && {
                        let rel = pose.inverse().compose(&p);
                        rel.t.norm() / depth <= params.seed_max_baseline
                            && rel.rotation_angle().to_degrees() <= params.seed_max_angle_deg
                    }
                {
                    dtam.add_to_keyframe(fr, p);
                    seeded += 1;
                }
            }
            self.kf_seeded = seeded;
            let publish_now = if params.publish_after_frames == 0 { 0 } else { usize::MAX };
            if let Some(kf) = dtam.solve_keyframe(publish_now).await {
                emit(SlamEvent::Keyframe(kf));
            }
            emit(SlamEvent::Stage("dense tracking".into()));
            self.last_kf_frame = f;
            self.last_range = range;
        }

        if std::env::var_os("DTAM_DEBUG").is_some() {
            let inter = prev.inverse().compose(&pose);
            eprintln!(
                "dbg {f}: final inter-frame rot {:.3}° trans {:.4} used {:.2} cov {:.2} attempts {}",
                inter.rotation_angle().to_degrees(),
                inter.t.norm(),
                stats_used,
                prediction.coverage,
                attempt
            );
        }
    }

    /// Finishes the active keyframe with everything it has accumulated.
    pub async fn finish(&mut self, emit: &mut impl FnMut(SlamEvent)) {
        if let Some(dtam) = self.dtam.as_mut()
            && dtam.active.as_ref().is_some_and(|a| a.frames_since_solve > 0)
            && let Some(kf) = dtam.solve_keyframe(0).await
        {
            emit(SlamEvent::Keyframe(kf));
        }
        emit(SlamEvent::Stage("done".into()));
    }
}
