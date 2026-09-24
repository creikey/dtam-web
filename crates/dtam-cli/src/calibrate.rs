//! Tracks a video and estimates the camera's focal length from the tracks.
//!
//! usage: dtam-calibrate [VIDEO] [--max-dim N] [--gaps 10,20,30] [--inlier-px 1.0] [--curve out.csv]

use std::path::PathBuf;

use dtam_core::calib::{FocalParams, estimate_focal};
use dtam_core::{Gpu, SlamPipeline, TrackerParams};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut video = PathBuf::from("test_video.mp4");
    let mut max_dim = 1024;
    let mut curve_out = None;
    let mut fp = FocalParams::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-dim" => max_dim = args.next().and_then(|v| v.parse().ok()).ok_or("--max-dim N")?,
            "--gaps" => {
                fp.pair_gaps = args
                    .next()
                    .ok_or("--gaps A,B,..")?
                    .split(',')
                    .map(|g| g.parse().map_err(|_| "bad --gaps"))
                    .collect::<Result<_, _>>()?;
            }
            "--inlier-px" => fp.inlier_px = args.next().and_then(|v| v.parse().ok()).ok_or("--inlier-px X")?,
            "--curve" => curve_out = Some(PathBuf::from(args.next().ok_or("--curve PATH")?)),
            _ => video = a.into(),
        }
    }

    let info = dtam_video::probe(&video)?;
    let (w, h) = dtam_video::scaled_size(&info, max_dim);
    let gpu = pollster::block_on(Gpu::headless())?;
    let mut pipeline = SlamPipeline::new(gpu, w, h, TrackerParams::default());
    let mut tracks = Vec::new();
    let t0 = std::time::Instant::now();
    dtam_video::decode(&video, (w, h), |frame| {
        tracks.push(pollster::block_on(pipeline.process(&frame)).tracks);
        true
    })?;
    eprintln!("tracked {} frames at {w}x{h} in {:.1}s", tracks.len(), t0.elapsed().as_secs_f32());

    let t0 = std::time::Instant::now();
    let est = estimate_focal(&tracks, w, h, &fp)?;
    eprintln!("calibrated in {:.1}s", t0.elapsed().as_secs_f32());

    let s = &est.stats;
    println!("frame pairs: {} tried, {} used", s.tried, s.used);
    println!(
        "  rejected: {} too few matches, {} too little motion, {} homography-like, {} fit failed",
        s.too_few_matches, s.too_little_flow, s.homography_like, s.fit_failed
    );
    let inlier_frac: f64 =
        est.pairs.iter().map(|p| p.f_inliers as f64 / p.matches as f64).sum::<f64>() / est.pairs.len() as f64;
    println!("  mean F inlier fraction: {:.1}%", inlier_frac * 100.0);
    println!();
    println!("focal length:     {:.1} px  (90% bootstrap: {:.1} .. {:.1})", est.focal_px, est.interval_px.0, est.interval_px.1);
    println!("per-pair median:  {:.1} px", est.per_pair_median_px);
    println!("horizontal FOV:   {:.2}°  ({w} px wide)", est.hfov_deg);
    println!("principal point:  ({:.1}, {:.1}) (assumed)", est.principal_point.0, est.principal_point.1);

    if let Some(path) = curve_out {
        let csv: String = std::iter::once("focal_px,cost\n".to_string())
            .chain(est.curve.iter().map(|(f, c)| format!("{f:.3},{c:.6}\n")))
            .collect();
        std::fs::write(&path, csv).map_err(|e| e.to_string())?;
        println!("cost curve written to {}", path.display());
    }
    Ok(())
}
