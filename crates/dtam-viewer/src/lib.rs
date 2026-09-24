//! DTAM viewer: native batch inspector (`app`) and the browser live app (`web`).

#[cfg(not(target_arch = "wasm32"))]
pub mod app;
pub mod ar;
pub mod scene3d;

#[cfg(target_arch = "wasm32")]
mod capture;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static RUNNER: std::cell::RefCell<Option<eframe::WebRunner>> = const { std::cell::RefCell::new(None) };
}

/// Forwards `log` records (wgpu / eframe errors included) to the console.
#[cfg(target_arch = "wasm32")]
struct ConsoleLog;

#[cfg(target_arch = "wasm32")]
impl log::Log for ConsoleLog {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn || m.target().starts_with("dtam")
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let msg = format!("[{}] {}: {}", r.level(), r.target(), r.args()).into();
        match r.level() {
            log::Level::Error => web_sys::console::error_1(&msg),
            log::Level::Warn => web_sys::console::warn_1(&msg),
            _ => web_sys::console::log_1(&msg),
        }
    }
    fn flush(&self) {}
}

/// Browser entry point: runs the live app in the canvas with id `canvas_id`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;
    static LOGGER: ConsoleLog = ConsoleLog;
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));
    let document = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
    let canvas = document
        .get_element_by_id(&canvas_id)
        .ok_or("canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    let runner = eframe::WebRunner::new();
    runner
        .start(canvas, eframe::WebOptions::default(), Box::new(|cc| Ok(Box::new(web::LiveApp::new(cc)))))
        .await?;
    RUNNER.with(|r| *r.borrow_mut() = Some(runner));
    Ok(())
}
