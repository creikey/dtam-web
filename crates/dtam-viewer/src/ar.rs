//! Viewer glue for the AR test cube (geometry lives in `dtam_core::ar`).

use dtam_core::geom::{Intrinsics, Se3};
use eframe::egui;

pub use dtam_core::ar::{Cube, anchor};

pub fn rasterize(
    cube: &Cube,
    t_wc: &Se3,
    k_video: &Intrinsics,
    out_w: u32,
    model_inv_depth: Option<(u32, u32, &[u16])>,
) -> Option<egui::ColorImage> {
    let (w, h, rgba) = dtam_core::ar::rasterize(cube, t_wc, k_video, out_w, model_inv_depth)?;
    Some(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba))
}
