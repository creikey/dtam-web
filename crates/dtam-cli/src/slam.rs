//! Runs the whole pipeline headless: KLT tracks, self-calibration, bootstrap,
//! DTAM mapping + tracking. Optionally writes the fused keyframe point cloud.
//!
//! usage: dtam-slam [VIDEO] [--frames N] [--ply out.ply] [--dump DIR] [--dump-every N]

use std::io::Write;
use std::path::PathBuf;

use dtam_core::slam::{PoseSource, Slam, SlamEvent, SlamParams};
use dtam_core::{Gpu, SlamPipeline, TrackerParams};

fn main() {
    if let Err(e) = run_cli() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_cli() -> Result<(), String> {
    let mut video = PathBuf::from("test_video.mp4");
    let mut ply = None;
    let mut dump: Option<PathBuf> = None;
    let mut max_frames = usize::MAX;
    let mut max_dim = 512u32;
    let mut dump_every = 50usize;
    let mut kf_window: Option<(usize, usize)> = None;
    let mut boot_frames: Option<usize> = None;
    let mut legacy_kf = false;
    let mut publish_after: Option<usize> = None;
    let mut seed_limit: Option<(f64, f64)> = None;
    let mut compare: Option<usize> = None;
    let mut segment: Option<(usize, usize)> = None;
    let mut coverage: Option<f32> = None;
    let mut nearby: Option<(f64, f64)> = None;
    let mut resolve: Option<usize> = None;
    let mut focal: Option<f64> = None;
    let mut render_ar: Option<(PathBuf, usize)> = None;
    let mut ref_seg: Option<(usize, usize)> = None;
    let mut render_3d: Option<PathBuf> = None;
    let mut track_iters: Option<Vec<u32>> = None;
    let mut min_used: Option<f32> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(args.next().ok_or("--ply PATH")?)),
            "--dump" => dump = Some(PathBuf::from(args.next().ok_or("--dump DIR")?)),
            "--dump-every" => dump_every = args.next().and_then(|v| v.parse().ok()).ok_or("--dump-every N")?,
            "--kf-window" => {
                let v = args.next().ok_or("--kf-window BEFORE,AFTER")?;
                let (a, b) = v.split_once(',').ok_or("--kf-window BEFORE,AFTER")?;
                kf_window = Some((a.parse().map_err(|_| "bad")?, b.parse().map_err(|_| "bad")?));
            }
            "--bootstrap-frames" => boot_frames = args.next().and_then(|v| v.parse().ok()),
            "--max-dim" => max_dim = args.next().and_then(|v| v.parse().ok()).ok_or("--max-dim N")?,
            "--compare" => compare = args.next().and_then(|v| v.parse().ok()),
            "--segment" => {
                let v = args.next().ok_or("--segment A,B")?;
                let (a, b) = v.split_once(',').ok_or("--segment A,B")?;
                segment = Some((a.parse().map_err(|_| "bad")?, b.parse().map_err(|_| "bad")?));
            }
            "--kf-coverage" => coverage = args.next().and_then(|v| v.parse().ok()),
            "--nearby" => {
                let v = args.next().ok_or("--nearby B,DEG")?;
                let (a, b) = v.split_once(',').ok_or("--nearby B,DEG")?;
                nearby = Some((a.parse().map_err(|_| "bad")?, b.parse().map_err(|_| "bad")?));
            }
            "--resolve-every" => resolve = args.next().and_then(|v| v.parse().ok()),
            "--min-used" => min_used = args.next().and_then(|v| v.parse().ok()),
            "--focal" => focal = args.next().and_then(|v| v.parse().ok()),
            "--render-ar" => {
                let dir = PathBuf::from(args.next().ok_or("--render-ar DIR EVERY")?);
                let every = args.next().and_then(|v| v.parse().ok()).ok_or("--render-ar DIR EVERY")?;
                render_ar = Some((dir, every));
            }
            "--ref-segment" => {
                let v = args.next().ok_or("--ref-segment A,B")?;
                let (a, b) = v.split_once(',').ok_or("--ref-segment A,B")?;
                ref_seg = Some((a.parse().map_err(|_| "bad")?, b.parse().map_err(|_| "bad")?));
            }
            "--track-iters" => track_iters = Some(args.next().ok_or("--track-iters a,b,c,d")?.split(',').map(|v| v.parse().unwrap_or(0)).collect()),
            "--render-3d" => render_3d = Some(PathBuf::from(args.next().ok_or("--render-3d DIR")?)),
            "--legacy-kf" => legacy_kf = true,
            "--publish-after" => publish_after = args.next().and_then(|v| v.parse().ok()),
            "--seed-limit" => {
                let v = args.next().ok_or("--seed-limit B,DEG")?;
                let (a, b) = v.split_once(',').ok_or("--seed-limit B,DEG")?;
                seed_limit = Some((a.parse().map_err(|_| "bad")?, b.parse().map_err(|_| "bad")?));
            }
            "--frames" => max_frames = args.next().and_then(|v| v.parse().ok()).ok_or("--frames N")?,
            _ => video = a.into(),
        }
    }

    let info = dtam_video::probe(&video)?;
    // The pipeline runs on the video scaled to `max_dim` (512 like the web app).
    let (w, h) = dtam_video::scaled_size(&info, max_dim);
    let gpu = pollster::block_on(Gpu::headless())?;
    let mut klt = SlamPipeline::new(gpu.clone(), w, h, TrackerParams::default());
    let (mut tracks, mut small) = (Vec::new(), Vec::new());
    let t0 = std::time::Instant::now();
    dtam_video::decode(&video, (w, h), |frame| {
        tracks.push(pollster::block_on(klt.process(&frame)).tracks);
        small.push(frame);
        tracks.len() < max_frames
    })?;
    eprintln!("tracked {} frames at {w}x{h} in {:.1}s", tracks.len(), t0.elapsed().as_secs_f32());

    let t0 = std::time::Instant::now();
    let mut keyframes = Vec::new();
    let mut sources = vec![None; tracks.len()];
    let mut stats = vec![None; tracks.len()];
    let mut intrinsics = None;
    let mut params = SlamParams::default();
    params.fixed_focal = focal;
    if let Some(t) = track_iters { params.dtam.track_iterations = t; }
    if let Some(r) = resolve { params.resolve_every = r; }
    if let Some(m) = min_used { params.min_used_for_mapping = m; }
    if let Some((b, a)) = nearby { params.max_keyframe_baseline = b; params.max_keyframe_angle_deg = a; }
    if let Some(c) = coverage { params.new_keyframe_coverage = c; }
    if let Some(b) = boot_frames { params.bootstrap_window = b; }
    if let Some(p) = publish_after { params.publish_after_frames = p; }
    if let Some((b, a)) = seed_limit { params.seed_max_baseline = b; params.seed_max_angle_deg = a; }
    if legacy_kf { params.publish_after_frames = 0; params.seed_max_baseline = f64::INFINITY; params.seed_max_angle_deg = f64::INFINITY; }
    if let Some(v) = kf_window { params.keyframe_frames_before = v.0; params.keyframe_frames_after = v.1; }
    if let Some((a, b)) = ref_seg {
        let f0 = dtam_core::calib::estimate_focal(&tracks, w, h, &Default::default()).map(|e| e.focal_px).unwrap_or(1400.0);
        let bp = dtam_core::sfm::BootstrapParams { max_frames: b - a, min_init_gap: 5, ..Default::default() };
        let seg = dtam_core::sfm::bootstrap(&tracks[a..b], dtam_core::geom::Intrinsics::centered(f0, w, h), &bp)?;
        eprintln!("reference BA {a}..{b}: rms {:.3} px, f {:.1}", seg.rms_reprojection_px, seg.intrinsics.fx);
        params.debug_reference = Some((a, seg.poses));
    }
    let mut slam = Slam::new(gpu, (w, h), params);
    let mut handler = |ev: SlamEvent| match ev {
            SlamEvent::Stage(s) => eprintln!("[{:6.1}s] {s}", t0.elapsed().as_secs_f32()),
            SlamEvent::Intrinsics { self_calibrated, refined, mapping } => { intrinsics = Some(refined); eprintln!(
                "intrinsics: self-calibrated f = {self_calibrated:.1}px, bundle-adjusted f = {:.1}px ({:.2}° HFOV), mapping {}x{} f = {:.1}px",
                refined.fx,
                refined.hfov_deg(),
                mapping.width,
                mapping.height,
                mapping.fx
            ) }
            SlamEvent::Pose { frame, pose, source } => sources[frame] = Some((source, pose)),
            SlamEvent::Tracking { frame, stats: s } => {
                if frame % 25 == 0 {
                    eprintln!(
                        "  frame {frame}: rmse {:.4} used {:.0}% rejected {:.0}% coverage {:.0}% iters {}",
                        s.rmse,
                        s.used_fraction * 100.0,
                        s.rejected_fraction * 100.0,
                        s.coverage * 100.0,
                        s.iterations
                    );
                }
                if let (Some(dir), Some((w, h, luma, depth)), Some((_, _, mask))) = (&dump, &s.prediction, &s.mask)
                    && frame % dump_every == 0
                {
                    let _ = std::fs::create_dir_all(dir);
                    let live = small[frame].downsample2();
                    let colors = [[40, 40, 200], [40, 220, 60], [240, 220, 40], [120, 120, 120]];
                    let mut out = Vec::new();
                    for y in 0..*h as usize {
                        for x in 0..*w as usize {
                            out.extend_from_slice(&live.rgb[(y * *w as usize + x) * 3..][..3]);
                        }
                        for x in 0..*w as usize {
                            let v = luma[y * *w as usize + x];
                            out.extend_from_slice(&[v, v, v]);
                        }
                        for x in 0..*w as usize {
                            let v = depth[y * *w as usize + x];
                            out.extend_from_slice(&[v, v, v]);
                        }
                        for x in 0..*w as usize {
                            out.extend_from_slice(&colors[mask[y * *w as usize + x].min(3) as usize]);
                        }
                    }
                    let path = dir.join(format!("track_{frame:04}.ppm"));
                    if let Ok(mut f) = std::fs::File::create(&path) {
                        let _ = write!(f, "P6\n{} {h}\n255\n", w * 4);
                        let _ = f.write_all(&out);
                    }
                }
                stats[frame] = Some(s);
            }
            SlamEvent::Keyframe(kf) => {
                let textured = kf.cost_range.iter().filter(|c| **c > 0.05).count() as f32 / kf.cost_range.len() as f32;
                eprintln!(
                    "  keyframe {} @ frame {}: {} frames, baseline {:.3}, xi [{:.3}, {:.3}], lambda {:.3}, {} iters, {:.0}% textured",
                    kf.id, kf.frame, kf.frames_used, kf.baseline, kf.xi_range.0, kf.xi_range.1, kf.lambda, kf.iterations, textured * 100.0
                );
                if kf.id < keyframes.len() {
                    let id = kf.id;
                    keyframes[id] = kf;
                } else {
                    keyframes.push(kf);
                }
            }
            SlamEvent::Phase(_) | SlamEvent::Klt { .. } => {}
    };
    for fr in &small {
        pollster::block_on(slam.push(fr, &mut handler));
    }
    pollster::block_on(slam.finish(&mut handler));
    drop(handler);

    let count = |s: PoseSource| sources.iter().filter(|x| x.is_some_and(|(src, _)| src == s)).count();
    println!(
        "poses: {} bootstrap, {} dense, {} predicted (tracking failed); {} keyframes; {:.1}s",
        count(PoseSource::Bootstrap),
        count(PoseSource::Dense),
        count(PoseSource::Predicted),
        keyframes.len(),
        t0.elapsed().as_secs_f32()
    );
    if let Some(Some((_, last))) = sources.iter().rev().find(|s| s.is_some()) {
        println!("final camera position: ({:.3}, {:.3}, {:.3})", last.t.x, last.t.y, last.t.z);
    }

    // Feature BA on a sub-segment: is the pinhole model consistent there, and
    // does dense tracking agree with it?
    if let Some((a, b)) = segment {
        use dtam_core::geom::Intrinsics;
        let k_used = sources.iter().flatten().next().map(|_| ());
        let _ = k_used;
        let f0 = dtam_core::calib::estimate_focal(&tracks, w, h, &Default::default()).map(|e| e.focal_px).unwrap_or(1400.0);
        for refine in [false, true] {
            let bp = dtam_core::sfm::BootstrapParams { max_frames: b - a, refine_focal: refine, min_init_gap: 5, ..Default::default() };
            match dtam_core::sfm::bootstrap(&tracks[a..b], Intrinsics::centered(f0, w, h), &bp) {
                Ok(seg) => {
                    println!(
                        "segment {a}..{b} (refine focal {refine}): BA rms {:.3} px over {} obs, f {:.1}",
                        seg.rms_reprojection_px, seg.observations, seg.intrinsics.fx
                    );
                    if refine {
                        continue;
                    }
                    println!("frame  BA rot from {a} (deg)  dense rot from {a} (deg)  diff (deg)");
                    let base = sources[a].map(|(_, p)| p);
                    for (i, q) in seg.poses.iter().enumerate().step_by(4) {
                        let f = a + i;
                        let ba_ang = q.rotation_angle().to_degrees();
                        if let (Some(p0), Some((_, p))) = (base, sources[f]) {
                            let rel = p0.inverse().compose(&p);
                            let diff = rel.r.transpose() * q.r;
                            let d = ((diff.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
                            println!("{f:5}  {ba_ang:8.3}  {:8.3}  {d:8.3}", rel.rotation_angle().to_degrees());
                        }
                    }
                }
                Err(e) => println!("segment BA failed: {e}"),
            }
        }
    }

    // Compare against a long feature bundle adjustment (reference trajectory).
    if let Some(nref) = compare {
        use dtam_core::geom::Intrinsics;
        let f0 = dtam_core::calib::estimate_focal(&tracks, w, h, &Default::default()).map(|e| e.focal_px).unwrap_or(1400.0);
        let bp = dtam_core::sfm::BootstrapParams { max_frames: nref, ..Default::default() };
        let reference = dtam_core::sfm::bootstrap(&tracks, Intrinsics::centered(f0, w, h), &bp)?;
        let n = reference.poses.len().min(sources.len());
        let pairs: Vec<(nalgebra::Vector3<f64>, nalgebra::Vector3<f64>)> =
            (0..n).filter_map(|i| sources[i].map(|(_, p)| (p.t, reference.poses[i].t))).collect();
        // Both trajectories use frame 0's camera as world; only scale differs.
        let (num, den) = pairs.iter().fold((0.0, 0.0), |(n, d), (x, y)| (n + x.dot(y), d + x.dot(x)));
        let scale = num / den.max(1e-12);
        let r = nalgebra::Matrix3::<f64>::identity();
        let t = nalgebra::Vector3::<f64>::zeros();
        println!("frame  src   pos err (x median depth)   rot err (deg)");
        for i in (0..n).step_by(5) {
            if let Some((src, p)) = sources[i] {
                let q = reference.poses[i];
                let pe = (r * p.t * scale + t - q.t).norm();
                let re = (r * p.r).transpose() * q.r;
                let ang = ((re.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
                println!("{i:5}  {:?}  {pe:.4}  {ang:.3}", src);
            }
        }
    }

    // Headless AR check: a cube anchored at frame 0's image center, drawn with
    // each frame's pose and occluded by the model's predicted depth.
    if let (Some((dir, every)), Some(k)) = (render_ar, intrinsics) {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let kf = keyframes.iter().min_by_key(|kf| kf.frame).ok_or("no keyframes")?;
        let p0 = sources[0].ok_or("no pose for frame 0")?.1;
        let center = [(k.width as f64 - 1.0) / 2.0, (k.height as f64 - 1.0) / 2.0];
        let cube = dtam_core::ar::anchor(kf, &p0, &k, center, 0.12).ok_or("could not anchor the cube")?;
        for f in (0..sources.len()).step_by(every.max(1)) {
            let Some((_, pose)) = sources[f] else { continue };
            let mut img = small[f].clone();
            let depth = stats[f].as_ref().and_then(|s: &dtam_core::dtam::TrackStats| s.prediction_inv_depth.as_ref());
            if let Some((_, _, rgba)) =
                dtam_core::ar::rasterize(&cube, &pose, &k, img.width, depth.map(|(w, h, d)| (*w, *h, d.as_slice())))
            {
                dtam_core::ar::composite(&mut img.rgb, &rgba);
            }
            let path = dir.join(format!("ar_{f:04}.ppm"));
            let mut out = std::fs::File::create(&path).map_err(|e| e.to_string())?;
            write!(out, "P6\n{} {}\n255\n", img.width, img.height).map_err(|e| e.to_string())?;
            out.write_all(&img.rgb).map_err(|e| e.to_string())?;
        }
        println!("wrote AR frames to {}", dir.display());
    }

    // Headless renders of the dense model as meshes.
    if let Some(dir) = render_3d {
        use dtam_core::viz::{MeshOptions, OrbitCamera, Shading, render_image};
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let gpu = pollster::block_on(Gpu::headless())?;
        let first = keyframes.first().ok_or("no keyframes")?;
        let base = OrbitCamera::framing(first);
        let views = [("front", base), ("side", OrbitCamera { yaw: base.yaw + 0.6, ..base }), ("top", OrbitCamera { pitch: base.pitch + 0.5, ..base })];
        let all: Vec<_> = keyframes.clone();
        let one = vec![first.clone()];
        for (set_name, set) in [("all", &all), ("kf0", &one)] {
            for (vname, cam) in &views {
                for (sname, shading, opts) in [
                    ("tex", Shading::Texture, MeshOptions::default()),
                    ("shaded", Shading::Shaded, MeshOptions::default()),
                    ("tex_c0", Shading::Texture, MeshOptions { min_confidence: 0.0, max_oblique_deg: 85.0, ..Default::default() }),
                    ("shaded_c0", Shading::Shaded, MeshOptions { min_confidence: 0.0, max_oblique_deg: 85.0, ..Default::default() }),
                ] {
                    let img = pollster::block_on(render_image(&gpu, set, &[], cam, [768, 576], shading, opts));
                    let rgb: Vec<u8> = img.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
                    let path = dir.join(format!("{set_name}_{vname}_{sname}.ppm"));
                    let mut f = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                    write!(f, "P6\n768 576\n255\n").map_err(|e| e.to_string())?;
                    f.write_all(&rgb).map_err(|e| e.to_string())?;
                }
            }
        }
        println!("wrote 3D renders to {}", dir.display());
    }

    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        for kf in &keyframes {
            let (w, h) = (kf.intrinsics.width, kf.intrinsics.height);
            let (lo, hi) = kf.xi_range;
            let gray = |v: &[f32]| -> Vec<u8> {
                v.iter().flat_map(|x| {
                    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
                    let g = (t * 255.0) as u8;
                    [g, g, g]
                }).collect()
            };
            for (name, data) in [("rgb", kf.rgb.clone()), ("argmin", gray(&kf.argmin_inv_depth)), ("depth", gray(&kf.inv_depth))] {
                let path = dir.join(format!("kf{}_{name}.ppm", kf.id));
                let mut f = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                write!(f, "P6\n{w} {h}\n255\n").map_err(|e| e.to_string())?;
                f.write_all(&data).map_err(|e| e.to_string())?;
            }
        }
    }

    if let Some(path) = ply {
        let pts: Vec<_> = keyframes.iter().flat_map(|k| k.points(2, 0.3)).collect();
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).map_err(|e| e.to_string())?);
        write!(
            f,
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
            pts.len()
        )
        .map_err(|e| e.to_string())?;
        for (p, c) in &pts {
            for v in p {
                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            f.write_all(c).map_err(|e| e.to_string())?;
        }
        println!("wrote {} points to {}", pts.len(), path.display());
    }
    Ok(())
}
