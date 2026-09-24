//! Browser frame sources: the bundled demo video (stepped frame by frame by
//! seeking) and the webcam. Frames are center-cropped to a square and resized
//! to `size`×`size` with a 2D canvas.

use dtam_core::Frame;
use js_sys::{Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{AddEventListenerOptions, CanvasRenderingContext2d, HtmlCanvasElement, HtmlVideoElement, MediaStream};

pub struct Capture {
    pub video: HtmlVideoElement,
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    stream: Option<MediaStream>,
    pub size: u32,
}

fn js_err(e: JsValue) -> String {
    e.as_string()
        .or_else(|| Reflect::get(&e, &"message".into()).ok().and_then(|m| m.as_string()))
        .unwrap_or_else(|| format!("{e:?}"))
}

/// Resolves when `target` fires `event` (once). The listener is registered
/// immediately, before the returned future is awaited.
pub fn once(target: &web_sys::EventTarget, event: &str) -> JsFuture {
    let target = target.clone();
    let event = event.to_string();
    let p = Promise::new(&mut |resolve, _| {
        let opts = AddEventListenerOptions::new();
        opts.set_once(true);
        let cb = Closure::once_into_js(move |_: JsValue| {
            let _ = resolve.call0(&JsValue::NULL);
        });
        let _ = target.add_event_listener_with_callback_and_add_event_listener_options(&event, cb.unchecked_ref(), &opts);
    });
    JsFuture::from(p)
}

/// Yields to the browser event loop for `ms` milliseconds.
pub async fn sleep(ms: i32) {
    let p = Promise::new(&mut |resolve, _| {
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
    });
    let _ = JsFuture::from(p).await;
}

impl Capture {
    fn new(size: u32) -> Result<Self, String> {
        let doc = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
        let video: HtmlVideoElement = doc.create_element("video").map_err(js_err)?.dyn_into().map_err(|_| "not a video")?;
        video.set_muted(true);
        video.set_attribute("playsinline", "").ok();
        video.set_attribute("muted", "").ok();
        video.set_cross_origin(Some("anonymous"));
        let canvas: HtmlCanvasElement = doc.create_element("canvas").map_err(js_err)?.dyn_into().map_err(|_| "not a canvas")?;
        canvas.set_width(size);
        canvas.set_height(size);
        let opts = Object::new();
        Reflect::set(&opts, &"willReadFrequently".into(), &true.into()).ok();
        let ctx: CanvasRenderingContext2d = canvas
            .get_context_with_context_options("2d", &opts)
            .map_err(js_err)?
            .ok_or("no 2d context")?
            .dyn_into()
            .map_err(|_| "not a 2d context")?;
        // Proper area filtering when the source is larger than `size`.
        ctx.set_image_smoothing_enabled(true);
        Reflect::set(&ctx, &"imageSmoothingQuality".into(), &"high".into()).ok();
        Ok(Self { video, canvas, ctx, stream: None, size })
    }

    /// The demo video at `url` (paused, ready to seek). It is downloaded
    /// into a Blob first so seeking works regardless of HTTP range support.
    pub async fn video_file(url: &str, size: u32) -> Result<Self, String> {
        let c = Self::new(size)?;
        let window = web_sys::window().ok_or("no window")?;
        let resp: web_sys::Response =
            JsFuture::from(window.fetch_with_str(url)).await.map_err(js_err)?.dyn_into().map_err(|_| "bad response")?;
        if !resp.ok() {
            return Err(format!("could not download {url} (HTTP {})", resp.status()));
        }
        let blob: web_sys::Blob =
            JsFuture::from(resp.blob().map_err(js_err)?).await.map_err(js_err)?.dyn_into().map_err(|_| "not a blob")?;
        let object_url = web_sys::Url::create_object_url_with_blob(&blob).map_err(js_err)?;
        c.video.set_preload("auto");
        c.video.set_src(&object_url);
        let ready = once(&c.video, "loadeddata");
        c.video.load();
        let _ = ready.await;
        if c.video.video_width() == 0 {
            return Err(format!("could not load {url}"));
        }
        Ok(c)
    }

    /// The (rear, if available) camera.
    pub async fn webcam(size: u32) -> Result<Self, String> {
        let mut c = Self::new(size)?;
        let devices = web_sys::window()
            .ok_or("no window")?
            .navigator()
            .media_devices()
            .map_err(|_| "camera access is not available here (needs https)")?;
        let video = Object::new();
        Reflect::set(&video, &"facingMode".into(), &"environment".into()).ok();
        let ideal = |v: u32| {
            let o = Object::new();
            Reflect::set(&o, &"ideal".into(), &v.into()).ok();
            o
        };
        Reflect::set(&video, &"width".into(), &ideal(1280)).ok();
        Reflect::set(&video, &"height".into(), &ideal(720)).ok();
        let constraints = web_sys::MediaStreamConstraints::new();
        constraints.set_video(&video);
        constraints.set_audio(&false.into());
        let stream: MediaStream = JsFuture::from(devices.get_user_media_with_constraints(&constraints).map_err(js_err)?)
            .await
            .map_err(|e| format!("camera: {}", js_err(e)))?
            .dyn_into()
            .map_err(|_| "not a media stream")?;
        c.video.set_src_object(Some(&stream));
        let ready = once(&c.video, "loadeddata");
        let _ = c.video.play();
        let _ = ready.await;
        c.stream = Some(stream);
        Ok(c)
    }

    pub fn duration(&self) -> f64 {
        self.video.duration()
    }

    /// Seeks the (paused) video and waits until the frame is ready.
    pub async fn seek(&self, t: f64) {
        let done = once(&self.video, "seeked");
        self.video.set_current_time(t);
        let _ = done.await;
    }

    /// Current video frame, center-cropped square, resized to `size`².
    pub fn grab(&self) -> Result<Frame, String> {
        let (vw, vh) = (self.video.video_width() as f64, self.video.video_height() as f64);
        if vw == 0.0 || vh == 0.0 {
            return Err("no video frame yet".into());
        }
        let s = vw.min(vh);
        let size = self.size as f64;
        self.ctx
            .draw_image_with_html_video_element_and_sw_and_sh_and_dx_and_dy_and_dw_and_dh(
                &self.video,
                (vw - s) / 2.0,
                (vh - s) / 2.0,
                s,
                s,
                0.0,
                0.0,
                size,
                size,
            )
            .map_err(js_err)?;
        let data = self.ctx.get_image_data(0.0, 0.0, size, size).map_err(js_err)?.data();
        let rgb: Vec<u8> = data.0.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        Ok(Frame::new(self.size, self.size, rgb))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if let Some(stream) = &self.stream {
            for t in stream.get_tracks().iter() {
                if let Ok(t) = t.dyn_into::<web_sys::MediaStreamTrack>() {
                    t.stop();
                }
            }
        }
        self.video.set_src_object(None);
        let _ = self.canvas.remove_attribute("id");
    }
}
