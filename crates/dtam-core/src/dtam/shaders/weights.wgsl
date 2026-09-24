// Edge-aware regulariser weight g(u) = exp(-alpha |grad I_r(u)|^beta) (paper eq. 5).

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<storage, read> ref_img: array<u32>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;

fn lum(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(p.w) - 1));
    let cy = u32(clamp(y, 0, i32(p.h) - 1));
    return luma(unpack_rgb(ref_img[cy * p.w + cx]));
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let gx = 0.5 * (lum(x + 1, y) - lum(x - 1, y));
    let gy = 0.5 * (lum(x, y + 1) - lum(x, y - 1));
    g[id.y * p.w + id.x] = exp(-p.alpha * pow(length(vec2f(gx, gy)), p.beta));
}
