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

    /// 2x2 box downsample (pixel-centered, matches `Intrinsics::scaled(0.5)`).
    pub fn downsample2(&self) -> Frame {
        let (w, h) = (self.width.div_ceil(2), self.height.div_ceil(2));
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut sum = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (2 * x + dx).min(self.width - 1);
                        let sy = (2 * y + dy).min(self.height - 1);
                        sum += self.rgb[((sy * self.width + sx) * 3 + c) as usize] as u32;
                    }
                    rgb[((y * w + x) * 3 + c) as usize] = ((sum + 2) / 4) as u8;
                }
            }
        }
        Frame { width: w, height: h, rgb }
    }

    /// Packed RGBA8 words (alpha = 255) for GPU upload.
    pub fn pack_rgba(&self) -> Vec<u32> {
        self.rgb
            .chunks_exact(3)
            .map(|c| c[0] as u32 | (c[1] as u32) << 8 | (c[2] as u32) << 16 | 0xff00_0000)
            .collect()
    }
}
