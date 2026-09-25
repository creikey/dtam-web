// Inter-frame rotation between consecutive live frames (paper §2.3.1,
// following Lovegrove & Davison): f_u = I_k(pi(K R K^-1 u)) - I_{k-1}(u),
// update R <- exp(w) R. Uses the first 6 + 3 accumulator slots.

@group(0) @binding(0) var<uniform> p: TrackParams;
@group(0) @binding(1) var<storage, read> tgt: array<f32>; // live pyramid (both slots)
@group(0) @binding(2) var<storage, read_write> partials: array<vec4f>;
@group(0) @binding(6) var<storage, read> st: GnState;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {
    var acc: array<vec4f, 9>;
    let w = p.dims.x;
    let h = p.dims.y;
    let npx = select(w * h, 0u, st.done != 0u);
    let t = st.cand;
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
        acc[7].y += 1.0;
        let r = sample_tgt(uv) - tgt[p.dims.w + pi];
        if (abs(r) > p.misc.x) {
            acc[7].x += 1.0;
            continue;
        }
        let a = image_jacobian(uv, xk);
        let j = cross(xk, a);
        acc[0] += j.x * vec4f(j.x, j.y, j.z, 0.0) + vec4f(0.0, 0.0, 0.0, j.y * j.y);
        acc[1] += vec4f(j.y * j.z, j.z * j.z, 0.0, 0.0);
        acc[5] += vec4f(0.0, j.x * r, j.y * r, j.z * r);
        acc[6].w += r * r;
        acc[7].z += 1.0;
    }
    reduce_and_store(acc, lid, wid.x);
}
