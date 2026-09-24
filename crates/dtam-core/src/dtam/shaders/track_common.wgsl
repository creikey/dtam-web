// Shared by the dense alignment passes. Each workgroup accumulates NACC sums
// over a grid-strided set of pixels and writes them to `partials`.
// Layout: [0..21) upper triangle of J^T J (row-major, i <= j),
//         [21..27) J^T r, 27 sum r^2, 28 rejected, 29 in view, 30 used,
//         31 pixels with a model prediction,
//         32..36 photometric sums over used pixels: I_l, I_v, I_l^2, I_l I_v.

struct TrackParams {
    t0: vec4f, // transform, column-major
    t1: vec4f,
    t2: vec4f,
    t3: vec4f,
    k: vec4f,     // fx fy cx cy at this level
    dims: vec4u,  // w, h, target offset, template offset
    misc: vec4f,  // outlier threshold, write mask (0/1), gain, bias
}

const WG: u32 = 64u;
const NACC: u32 = 36u;
var<workgroup> sh: array<f32, 2304>; // WG * NACC

fn tgt_at(x: u32, y: u32) -> f32 {
    return tgt[p.dims.z + y * p.dims.x + x];
}

fn sample_tgt(uv: vec2f) -> f32 {
    let f = floor(uv);
    let t = uv - f;
    let x0 = u32(f.x);
    let y0 = u32(f.y);
    let x1 = min(x0 + 1u, p.dims.x - 1u);
    let y1 = min(y0 + 1u, p.dims.y - 1u);
    return mix(mix(tgt_at(x0, y0), tgt_at(x1, y0), t.x), mix(tgt_at(x0, y1), tgt_at(x1, y1), t.x), t.y);
}

// Photometric gradient chained through the projection: dI/dX at camera point x.
fn image_jacobian(uv: vec2f, x: vec3f) -> vec3f {
    let gx = 0.5 * (sample_tgt(uv + vec2f(1.0, 0.0)) - sample_tgt(uv - vec2f(1.0, 0.0)));
    let gy = 0.5 * (sample_tgt(uv + vec2f(0.0, 1.0)) - sample_tgt(uv - vec2f(0.0, 1.0)));
    let iz = 1.0 / x.z;
    return vec3f(gx * p.k.x * iz, gy * p.k.y * iz, -(gx * p.k.x * x.x + gy * p.k.y * x.y) * iz * iz);
}

fn reduce_and_store(acc: array<f32, NACC>, lid: u32, wid: u32) {
    for (var k = 0u; k < NACC; k++) {
        sh[lid * NACC + k] = acc[k];
    }
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            for (var k = 0u; k < NACC; k++) {
                sh[lid * NACC + k] += sh[(lid + s) * NACC + k];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        for (var k = 0u; k < NACC; k++) {
            partials[wid * NACC + k] = sh[k];
        }
    }
}
