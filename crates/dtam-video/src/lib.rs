//! Native video decoding by piping raw RGB frames out of `ffmpeg`.
//! (On wasm this will be replaced by WebCodecs / an HTMLVideoElement.)

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use dtam_core::Frame;

pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub frames: Option<usize>,
    pub fps: f32,
}

pub fn probe(path: &Path) -> Result<VideoInfo, String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,nb_frames,r_frame_rate"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .map_err(|e| format!("failed to run ffprobe (is ffmpeg installed?): {e}"))?;
    if !out.status.success() {
        return Err(format!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let field = |k: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(k).and_then(|l| l.strip_prefix('=')))
            .map(str::trim)
    };
    let width = field("width").and_then(|v| v.parse().ok()).ok_or("no video width")?;
    let height = field("height").and_then(|v| v.parse().ok()).ok_or("no video height")?;
    let frames = field("nb_frames").and_then(|v| v.parse().ok());
    let fps = field("r_frame_rate")
        .and_then(|v| {
            let (n, d) = v.split_once('/')?;
            Some(n.parse::<f32>().ok()? / d.parse::<f32>().ok()?)
        })
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or(30.0);
    Ok(VideoInfo { width, height, frames, fps })
}

/// Output size after limiting the longest side to `max_dim` (kept even).
pub fn scaled_size(info: &VideoInfo, max_dim: u32) -> (u32, u32) {
    let s = (max_dim as f32 / info.width.max(info.height) as f32).min(1.0);
    let even = |v: u32| ((v as f32 * s).round() as u32 / 2 * 2).max(2);
    (even(info.width), even(info.height))
}

/// Decodes every frame at `size`, calling `on_frame` until it returns false.
pub fn decode(
    path: &Path,
    size: (u32, u32),
    mut on_frame: impl FnMut(Frame) -> bool,
) -> Result<(), String> {
    let (w, h) = size;
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-vf", &format!("scale={w}:{h}:flags=area")])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to run ffmpeg: {e}"))?;
    let mut stdout = child.stdout.take().unwrap();
    let frame_bytes = (w * h * 3) as usize;
    let mut stopped = false;
    loop {
        let mut buf = vec![0u8; frame_bytes];
        match stdout.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("reading ffmpeg output: {e}")),
        }
        if !on_frame(Frame::new(w, h, buf)) {
            let _ = child.kill();
            stopped = true;
            break;
        }
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !stopped && !out.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(())
}
