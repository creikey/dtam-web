//! Camera geometry: SE(3) poses and pinhole intrinsics.
//!
//! Conventions (as in the DTAM paper): camera frame has x right, y down,
//! z forward; `T_wc` maps camera points into the world, `x_w = R x_c + t`.
//! Pixel centers sit at integer coordinates.

use nalgebra::{Matrix3, Vector3};

pub type Vec3 = Vector3<f64>;
pub type Mat3 = Matrix3<f64>;

/// Rigid transform `x' = r x + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3 {
    pub r: Mat3,
    pub t: Vec3,
}

impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Se3 {
    pub fn identity() -> Self {
        Self { r: Mat3::identity(), t: Vec3::zeros() }
    }

    pub fn new(r: Mat3, t: Vec3) -> Self {
        Self { r, t }
    }

    pub fn inverse(&self) -> Self {
        let rt = self.r.transpose();
        Self { r: rt, t: -(rt * self.t) }
    }

    pub fn transform(&self, p: &Vec3) -> Vec3 {
        self.r * p + self.t
    }

    /// `self ∘ other`: apply `other` first.
    pub fn compose(&self, other: &Se3) -> Se3 {
        Se3 { r: self.r * other.r, t: self.r * other.t + self.t }
    }

    /// Exponential map of a twist `(v, w)` (translation part first).
    pub fn exp(xi: &[f64; 6]) -> Se3 {
        let v = Vec3::new(xi[0], xi[1], xi[2]);
        let w = Vec3::new(xi[3], xi[4], xi[5]);
        let theta = w.norm();
        let wx = skew(&w);
        let (a, b, c) = if theta < 1e-8 {
            (1.0 - theta * theta / 6.0, 0.5 - theta * theta / 24.0, 1.0 / 6.0 - theta * theta / 120.0)
        } else {
            let (s, co) = theta.sin_cos();
            (s / theta, (1.0 - co) / (theta * theta), (theta - s) / (theta * theta * theta))
        };
        let wx2 = wx * wx;
        let r = Mat3::identity() + wx * a + wx2 * b;
        let jl = Mat3::identity() + wx * b + wx2 * c;
        Se3 { r: orthonormalize(&r), t: jl * v }
    }

    /// Rotation-only exponential (Rodrigues).
    pub fn exp_rot(w: &Vec3) -> Mat3 {
        Se3::exp(&[0.0, 0.0, 0.0, w.x, w.y, w.z]).r
    }

    pub fn rotation_angle(&self) -> f64 {
        ((self.r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
    }

    /// Column-major 4x4 as f32, for GPU uniforms.
    pub fn to_mat4_f32(&self) -> [[f32; 4]; 4] {
        let r = &self.r;
        [
            [r[(0, 0)] as f32, r[(1, 0)] as f32, r[(2, 0)] as f32, 0.0],
            [r[(0, 1)] as f32, r[(1, 1)] as f32, r[(2, 1)] as f32, 0.0],
            [r[(0, 2)] as f32, r[(1, 2)] as f32, r[(2, 2)] as f32, 0.0],
            [self.t.x as f32, self.t.y as f32, self.t.z as f32, 1.0],
        ]
    }
}

pub fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Nearest rotation matrix (via SVD).
pub fn orthonormalize(m: &Mat3) -> Mat3 {
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        let mut u2 = u;
        u2.column_mut(2).neg_mut();
        r = u2 * vt;
    }
    r
}

/// Pinhole intrinsics (square pixels, zero skew allowed to differ in fx/fy).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// Square pixels, principal point at the image center.
    pub fn centered(f: f64, width: u32, height: u32) -> Self {
        Self {
            fx: f,
            fy: f,
            cx: (width as f64 - 1.0) / 2.0,
            cy: (height as f64 - 1.0) / 2.0,
            width,
            height,
        }
    }

    pub fn k(&self) -> Mat3 {
        Mat3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn k_inv(&self) -> Mat3 {
        Mat3::new(
            1.0 / self.fx,
            0.0,
            -self.cx / self.fx,
            0.0,
            1.0 / self.fy,
            -self.cy / self.fy,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Intrinsics of the image resized by `s` (pixel-centered convention).
    pub fn scaled(&self, s: f64) -> Self {
        Self {
            fx: self.fx * s,
            fy: self.fy * s,
            cx: (self.cx + 0.5) * s - 0.5,
            cy: (self.cy + 0.5) * s - 0.5,
            width: (self.width as f64 * s).round() as u32,
            height: (self.height as f64 * s).round() as u32,
        }
    }

    /// Intrinsics of pyramid level `l` built with 2x pixel-centered decimation.
    pub fn level(&self, l: u32) -> Self {
        let s = 1.0 / (1u32 << l) as f64;
        Self {
            fx: self.fx * s,
            fy: self.fy * s,
            cx: (self.cx + 0.5) * s - 0.5,
            cy: (self.cy + 0.5) * s - 0.5,
            width: self.width.div_ceil(1 << l),
            height: self.height.div_ceil(1 << l),
        }
    }

    pub fn project(&self, p: &Vec3) -> [f64; 2] {
        [self.fx * p.x / p.z + self.cx, self.fy * p.y / p.z + self.cy]
    }

    /// Ray through pixel `u` at unit depth (z = 1).
    pub fn unproject(&self, u: [f64; 2]) -> Vec3 {
        Vec3::new((u[0] - self.cx) / self.fx, (u[1] - self.cy) / self.fy, 1.0)
    }

    pub fn in_bounds(&self, u: [f64; 2], margin: f64) -> bool {
        u[0] >= margin
            && u[1] >= margin
            && u[0] <= self.width as f64 - 1.0 - margin
            && u[1] <= self.height as f64 - 1.0 - margin
    }

    pub fn hfov_deg(&self) -> f64 {
        2.0 * (self.width as f64 / (2.0 * self.fx)).atan().to_degrees()
    }

    /// `[fx, fy, cx, cy]` as f32 for GPU uniforms.
    pub fn to_f32(&self) -> [f32; 4] {
        [self.fx as f32, self.fy as f32, self.cx as f32, self.cy as f32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_rotation_and_inverse() {
        let t = Se3::exp(&[0.3, -0.2, 0.5, 0.1, -0.4, 0.25]);
        assert!((t.r.transpose() * t.r - Mat3::identity()).norm() < 1e-12);
        let p = Vec3::new(1.0, 2.0, 3.0);
        let q = t.inverse().transform(&t.transform(&p));
        assert!((p - q).norm() < 1e-12);
    }

    #[test]
    fn scaled_intrinsics_are_pixel_centered() {
        let k = Intrinsics::centered(1000.0, 1024, 1024);
        let h = k.scaled(0.5);
        assert_eq!((h.width, h.cx), (512, 255.5));
        let p = Vec3::new(0.1, -0.2, 2.0);
        let (u0, u1) = (k.project(&p), h.project(&p));
        assert!(((u0[0] + 0.5) * 0.5 - 0.5 - u1[0]).abs() < 1e-9);
    }
}
