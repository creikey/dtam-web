// Inter-frame rotation between consecutive live frames (paper §2.3.1,
// following Lovegrove & Davison): f_u = I_k(pi(K R K^-1 u)) - I_{k-1}(u),
// update R <- exp(w) R. Uses the first 6 + 3 accumulator slots.

@group(0) @binding(0) var<uniform> p: TrackParams;
@group(0) @binding(1) var<storage, read> tgt: array<f32>; // live pyramid (both slots)
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {
    var acc: array<f32, NACC>;
    let w = p.dims.x;
    let h = p.dims.y;
    let npx = w * h;
    let t = mat4x4f(p.t0, p.t1, p.t2, p.t3);
    let maxc = vec2f(f32(w) - 2.0, f32(h) - 2.0);
    for (var pi = wid.x * WG + lid; pi < npx; pi += nwg.x * WG) {
        let x = pi % w;
        let y = pi / w;
        let ray = vec3f((f32(x) - p.k.z) / p.k.x, (f32(y) - p.k.w) / p.k.y, 1.0);
        let xk = (t * vec4f(ray, 0.0)).xyz;
        if (xk.z <= 1e-6) {
            continue;
        }
        let uv = vec2f(p.k.x * xk.x / xk.z + p.k.z, p.k.y * xk.y / xk.z + p.k.w);
        if (any(uv < vec2f(1.0)) || any(uv > maxc)) {
            continue;
        }
        acc[29] += 1.0;
        let r = sample_tgt(uv) - tgt[p.dims.w + pi];
        if (abs(r) > p.misc.x) {
            acc[28] += 1.0;
            continue;
        }
        let a = image_jacobian(uv, xk);
        let j = cross(xk, a);
        acc[0] += j.x * j.x;
        acc[1] += j.x * j.y;
        acc[2] += j.x * j.z;
        acc[3] += j.y * j.y;
        acc[4] += j.y * j.z;
        acc[5] += j.z * j.z;
        acc[21] += j.x * r;
        acc[22] += j.y * r;
        acc[23] += j.z * r;
        acc[27] += r * r;
        acc[30] += 1.0;
    }
    reduce_and_store(acc, lid, wid.x);
}
