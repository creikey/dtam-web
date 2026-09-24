/// One decoded video frame, 8-bit sRGB, tightly packed RGB rows.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl Frame {
    pub fn new(width: u32, height: u32, rgb: Vec<u8>) -> Self {
        assert_eq!(rgb.len(), (width * height * 3) as usize, "frame size mismatch");
        Self { width, height, rgb }
    }

    /// Rec.709 luma of the gamma-encoded values (no linearization; photometric
    /// tracking only needs a consistent intensity, not physical radiance).
    pub fn luma(&self) -> Vec<u8> {
        self.rgb
            .chunks_exact(3)
            .map(|c| {
                let y = 0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32;
                (y + 0.5) as u8
            })
            .collect()
    }
}
