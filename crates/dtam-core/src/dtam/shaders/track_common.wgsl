// Shared by the dense alignment passes. Each workgroup accumulates NACC sums
// over a grid-strided set of pixels and writes them to `partials`.
// Layout: [0..21) upper triangle of J^T J (row-major, i <= j),
//         [21..27) J^T r, 27 sum r^2, 28 rejected, 29 in view, 30 used,
//         31 pixels with a model prediction,
//         32..36 photometric sums over used pixels: I_l, I_v, I_l^2, I_l I_v.

const WG: u32 = 64u;
// Accumulators as 9 vec4s (indices 4k..4k+3), kept in registers with
// constant indexing; reduced through workgroup memory laid out [k][lid].
var<workgroup> sh: array<vec4f, 576>; // 9 * WG

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

fn reduce_and_store(acc: array<vec4f, 9>, lid: u32, wid: u32) {
    for (var k = 0u; k < 9u; k++) {
        sh[k * WG + lid] = acc[k];
    }
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            for (var k = 0u; k < 9u; k++) {
                sh[k * WG + lid] += sh[k * WG + lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lid < 9u) {
        partials[wid * 9u + lid] = sh[lid * WG];
    }
}
