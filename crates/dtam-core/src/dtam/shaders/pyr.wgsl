// Grayscale pyramids for dense tracking.
// unpack: packed RGBA8 -> luma f32 at level 0.
// down:   2x2 box average (pixel-centered, matches Intrinsics::level);
//         mode 1 averages only values > 0 (inverse depth, 0 = no surface).

struct PyrParams {
    src_off: u32,
    src_w: u32,
    src_h: u32,
    mode: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: PyrParams;
@group(0) @binding(1) var<storage, read> rgba: array<u32>;
@group(0) @binding(2) var<storage, read_write> pyr: array<f32>;

@compute @workgroup_size(16, 16)
fn unpack(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.dst_w || id.y >= p.dst_h) {
        return;
    }
    let i = id.y * p.dst_w + id.x;
    let c = rgba[i];
    let rgb = vec3f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
    pyr[p.dst_off + i] = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.dst_w || id.y >= p.dst_h) {
        return;
    }
    var sum = 0.0;
    var n = 0.0;
    for (var j = 0u; j < 2u; j++) {
        for (var i = 0u; i < 2u; i++) {
            let x = min(id.x * 2u + i, p.src_w - 1u);
            let y = min(id.y * 2u + j, p.src_h - 1u);
            let v = pyr[p.src_off + y * p.src_w + x];
            if (p.mode == 0u || v > 0.0) {
                sum += v;
                n += 1.0;
            }
        }
    }
    pyr[p.dst_off + id.y * p.dst_w + id.x] = select(0.0, sum / n, n > 0.0);
}
