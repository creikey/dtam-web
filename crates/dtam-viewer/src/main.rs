mod app;
#[cfg(not(target_arch = "wasm32"))]
mod video;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init();
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n\nusage: dtam-viewer [VIDEO] [--max-dim N]");
            std::process::exit(2);
        }
    };
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 950.0])
            .with_title("DTAM viewer"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "dtam-viewer",
        options,
        Box::new(move |cc| Ok(Box::new(app::ViewerApp::new_native(cc, args.video, args.max_dim)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Web entry point (eframe::WebRunner + WebCodecs frame source) not wired up yet.
}

#[cfg(not(target_arch = "wasm32"))]
struct Args {
    video: std::path::PathBuf,
    max_dim: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl Args {
    fn parse() -> Result<Self, String> {
        let mut video = None;
        let mut max_dim = 1024;
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--max-dim" => {
                    max_dim = it
                        .next()
                        .and_then(|v| v.parse().ok())
                        .ok_or("--max-dim needs a positive integer")?;
                }
                "-h" | "--help" => return Err("DTAM viewer".into()),
                _ if video.is_none() => video = Some(a.into()),
                _ => return Err(format!("unexpected argument {a:?}")),
            }
        }
        Ok(Self {
            video: video.unwrap_or_else(|| "test_video.mp4".into()),
            max_dim,
        })
    }
}
