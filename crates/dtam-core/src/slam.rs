//! Offline sequence driver: tracks -> self-calibration -> feature bootstrap ->
//! first DTAM keyframe -> dense tracking with coverage-triggered keyframes.

use std::collections::HashMap;
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
    /// Reference frame of the first keyframe (default: middle of the
    /// bootstrap window, using the whole window as I(r)).
    pub first_keyframe: Option<usize>,
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
            max_keyframe_baseline: f64::INFINITY,
            max_keyframe_angle_deg: f64::INFINITY,
            min_used_for_mapping: 0.8,
            new_keyframe_coverage: 0.92,
            min_keyframe_spacing: 8,
            xi_margin: (0.5, 1.6),
            max_depth_ratio: 36.0,
            first_keyframe: None,
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

#[derive(Clone, Debug)]
pub enum SlamEvent {
    Stage(String),
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

/// Runs the full pipeline. `tracks` are KLT tracks at `video_size`;
/// `frame(i)` returns video frame `i` at the DTAM mapping resolution (any
/// uniform downscale of the video, e.g. half size).
pub async fn run(
    gpu: Gpu,
    tracks: &[FrameTracks],
    video_size: (u32, u32),
    mut frame: impl FnMut(usize) -> Frame,
    params: &SlamParams,
    mut emit: impl FnMut(SlamEvent),
) -> Result<(), String> {
    let n = tracks.len();
    let (w, h) = video_size;

    emit(SlamEvent::Stage("self-calibrating focal length".into()));
    let f0 = match params.fixed_focal.map(Ok).unwrap_or_else(|| estimate_focal(tracks, w, h, &params.focal).map(|e| e.focal_px)) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("focal self-calibration failed ({e}); assuming 60° FOV");
            w as f64 / 2.0 / (30f64).to_radians().tan()
        }
    };

    emit(SlamEvent::Stage("bootstrapping with features".into()));
    let bp = BootstrapParams { refine_focal: params.bootstrap.refine_focal && params.fixed_focal.is_none(), ..params.bootstrap.clone() };
    let boot = bootstrap(tracks, Intrinsics::centered(f0, w, h), &bp)?;
    let first = frame(0);
    let k_map = boot.intrinsics.scaled(first.width as f64 / w as f64);
    emit(SlamEvent::Intrinsics { self_calibrated: f0, refined: boot.intrinsics, mapping: k_map });
    let n0 = boot.poses.len();
    let mut poses: Vec<Option<Se3>> = vec![None; n];
    for (i, p) in boot.poses.iter().enumerate() {
        poses[i] = Some(*p);
        emit(SlamEvent::Pose { frame: i, pose: *p, source: PoseSource::Bootstrap });
    }

    let mut cache: HashMap<usize, Frame> = HashMap::from([(0, first)]);
    let mut small = |i: usize, cache: &mut HashMap<usize, Frame>| -> Frame {
        cache.entry(i).or_insert_with(|| frame(i)).clone()
    };

    let mut dtam = Dtam::new(gpu, k_map, params.dtam.clone());

    // First keyframe in the middle of the bootstrap window, using all of it.
    let r0 = params.first_keyframe.unwrap_or(n0 / 2).min(n0 - 1);
    let window = if params.first_keyframe.is_some() {
        r0.saturating_sub(params.keyframe_frames_before)..(r0 + params.keyframe_frames_after + 1).min(n0)
    } else {
        0..n0
    };
    emit(SlamEvent::Stage(format!("mapping keyframe 0 (frame {r0})")));
    let t_rw = boot.poses[r0].inverse();
    let mut xis: Vec<f32> = boot
        .points
        .iter()
        .map(|p| t_rw.transform(&p.pos).z)
        .filter(|z| *z > 1e-3)
        .map(|z| 1.0 / z as f32)
        .collect();
    xis.sort_by(f32::total_cmp);
    if xis.is_empty() {
        return Err("no bootstrap points in front of the first keyframe".into());
    }
    let pct = |v: &[f32], p: f32| v[((v.len() - 1) as f32 * p) as usize];
    let range0 = (pct(&xis, 0.02) * params.xi_margin.0, pct(&xis, 0.98) * params.xi_margin.1);
    let reference = small(r0, &mut cache);
    let others: Vec<(Frame, Se3)> = window.filter(|&i| i != r0).map(|i| (small(i, &mut cache), boot.poses[i])).collect();
    let refs: Vec<(&Frame, Se3)> = others.iter().map(|(f, p)| (f, *p)).collect();
    // The first keyframe stays active: every later tracked frame keeps being
    // added to its cost volume until the next keyframe replaces it.
    dtam.begin_keyframe(r0, &reference, boot.poses[r0], range0, 1.0);
    for (f, p) in &refs {
        dtam.add_to_keyframe(f, *p);
    }
    if let Some(kf) = dtam.solve_keyframe().await {
        emit(SlamEvent::Keyframe(kf));
    }
    // Model predictions for the bootstrap frames too (display / occlusion).
    for (i, p) in boot.poses.iter().enumerate() {
        let pred = dtam.predict(*p).await;
        emit(SlamEvent::Tracking { frame: i, stats: pred.to_stats() });
    }
    let mut last_kf_frame = r0;
    let mut last_range = range0;
    let mut tracked = vec![false; n];
    tracked[..n0].fill(true);
    let mut last_good = boot.poses[n0 - 1];
    let mut ref_anchor: Option<(Se3, f64)> = None;

    // Dense tracking from the end of the bootstrap window.
    emit(SlamEvent::Stage("dense tracking".into()));
    if n0 > 0 {
        dtam.set_live(&small(n0 - 1, &mut cache), (n0 - 1) % 2);
    }
    for f in n0..n {
        let slot = f % 2;
        let live = small(f, &mut cache);
        dtam.set_live(&live, slot);
        let r_cur_prev = dtam.rotation(1 - slot, slot).await;

        let prev = poses[f - 1].unwrap();
        let prev2 = if f >= 2 { poses[f - 2].unwrap_or(prev) } else { prev };
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
                candidates.push(last_good);
            }
            prediction.get_or_insert(pred);
            attempt += 1;
        }
        let prediction = prediction.unwrap();
        let (pose, source, stats) = match result {
            Some((pose, stats)) => (pose, PoseSource::Dense, stats),
            None => (predicted, PoseSource::Predicted, TrackStats { coverage: prediction.coverage, ..Default::default() }),
        };
        if source == PoseSource::Dense {
            last_good = pose;
        }
        if std::env::var_os("DTAM_DEBUG").is_some() {
            let inter = prev.inverse().compose(&pose);
            let pre = crate::geom::Se3::new(r_cur_prev, crate::geom::Vec3::zeros()).rotation_angle().to_degrees();
            eprintln!(
                "dbg {f}: prealign {pre:.3}° final inter-frame rot {:.3}° trans {:.4} used {:.2} rmse {:.4} cov {:.2} attempts {}",
                inter.rotation_angle().to_degrees(),
                inter.t.norm(),
                stats.used_fraction,
                stats.rmse,
                prediction.coverage,
                attempt
            );
        }
        if let Some((a, refs)) = &params.debug_reference
            && f >= *a
            && f - a < refs.len()
        {
            if f == *a {
                let p50 = prediction.inv_depth_percentiles[1].max(1e-3) as f64;
                ref_anchor = Some((pose, 1.0 / p50));
            }
            if let Some((anchor, scale)) = ref_anchor {
                let seg = refs[f - a];
                let reference = anchor.compose(&Se3::new(seg.r, seg.t * scale));
                let (c_ref, u_ref) = dtam.cost(slot, &prediction, reference).await;
                let (c_trk, u_trk) = dtam.cost(slot, &prediction, pose).await;
                let (c_pred, _) = dtam.cost(slot, &prediction, prediction.pose).await;
                let d = reference.inverse().compose(&pose);
                eprintln!(
                    "ref {f}: cost ref {c_ref:.5} (used {u_ref:.2}) tracked {c_trk:.5} (used {u_trk:.2}) start {c_pred:.5} | tracked vs ref: rot {:.2}° trans {:.4}",
                    d.rotation_angle().to_degrees(),
                    d.t.norm()
                );
            }
        }
        poses[f] = Some(pose);
        tracked[f] = source == PoseSource::Dense;
        emit(SlamEvent::Pose { frame: f, pose, source });
        let stats_used = stats.used_fraction;
        emit(SlamEvent::Tracking { frame: f, stats });

        // Mapping (paper §2.2.1): the active keyframe keeps averaging in every
        // tracked frame, and its regularised solution is refreshed regularly.
        // I(r) = frames "nearby and overlapping r": bounded baseline and
        // rotation from the keyframe, and confidently tracked.
        let nearby = dtam.keyframes.last().zip(dtam.active.as_ref()).is_some_and(|(kf, akf)| {
            let rel = akf.t_wr.inverse().compose(&pose);
            rel.t.norm() * kf.median_inv_depth as f64 <= params.max_keyframe_baseline
                && rel.rotation_angle().to_degrees() <= params.max_keyframe_angle_deg
        });
        if source == PoseSource::Dense && nearby && stats_used >= params.min_used_for_mapping {
            dtam.add_to_keyframe(&live, pose);
            if dtam.active.as_ref().is_some_and(|a| a.frames_since_solve >= params.resolve_every) {
                if let Some(kf) = dtam.solve_keyframe().await {
                    emit(SlamEvent::Keyframe(kf));
                }
            }
        }

        // Keyframe management (paper §2.4): new keyframe when the model no
        // longer covers enough of the view.
        if source == PoseSource::Dense
            && f - last_kf_frame >= params.min_keyframe_spacing
            && prediction.coverage < params.new_keyframe_coverage
        {
            let [p2, p50, p98] = prediction.inv_depth_percentiles;
            // The paper does not say how D = [ξmin, ξmax] is chosen; we span the
            // predicted inverse depths (with margins), capped around the median.
            let range = if p2 > 0.0 && p98 > p2 {
                let spread = params.max_depth_ratio.sqrt();
                ((p2 * params.xi_margin.0).max(p50 / spread), (p98 * params.xi_margin.1).min(p50 * spread))
            } else {
                last_range
            };
            // λ = 1 / (1 + 0.5 d_min), d_min = nearest predicted depth (§2.2.6).
            let min_depth = params.xi_margin.1 / range.1;
            let lambda = 1.0 / (1.0 + 0.5 * min_depth);

            // Finish the outgoing keyframe with everything it has accumulated.
            if dtam.active.as_ref().is_some_and(|a| a.frames_since_solve > 0)
                && let Some(kf) = dtam.solve_keyframe().await
            {
                emit(SlamEvent::Keyframe(kf));
            }
            emit(SlamEvent::Stage(format!("mapping keyframe {} (frame {f})", dtam.keyframes.len())));
            dtam.begin_keyframe(f, &live, pose, range, lambda);
            // Seed it with the already tracked frames just before it that are
            // nearby (same rule as below).
            let depth = 2.0 / (range.0 / params.xi_margin.0 + range.1 / params.xi_margin.1) as f64;
            for i in f.saturating_sub(params.keyframe_frames_before)..f {
                if tracked[i]
                    && let Some(p) = poses[i]
                    && {
                        let rel = pose.inverse().compose(&p);
                        rel.t.norm() / depth <= params.max_keyframe_baseline
                            && rel.rotation_angle().to_degrees() <= params.max_keyframe_angle_deg
                    }
                {
                    let fr = small(i, &mut cache);
                    dtam.add_to_keyframe(&fr, p);
                }
            }
            if let Some(kf) = dtam.solve_keyframe().await {
                emit(SlamEvent::Keyframe(kf));
            }
            emit(SlamEvent::Stage("dense tracking".into()));
            last_kf_frame = f;
            last_range = range;
        }

        let keep_from = f.saturating_sub(params.keyframe_frames_before + 2);
        cache.retain(|&i, _| i >= keep_from);
    }
    if dtam.active.as_ref().is_some_and(|a| a.frames_since_solve > 0)
        && let Some(kf) = dtam.solve_keyframe().await
    {
        emit(SlamEvent::Keyframe(kf));
    }
    emit(SlamEvent::Stage("done".into()));
    Ok(())
}
