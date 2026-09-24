// 6DOF forward-compositional alignment of the live image to the model
// prediction (paper eqs. 19-21): f_u = I_l(pi(K T_lv exp(psi) pi^-1(u, xi_v(u)))) - I_v(u),
// with a global gain/bias on I_l (a I_l + b) for robustness to exposure
// changes (the affine invariance NCC gives, paper §3.2 / [5]).
// Pixels with |f_u| above a threshold are rejected (paper §2.3.2).

@group(0) @binding(0) var<uniform> p: TrackParams;
@group(0) @binding(1) var<storage, read> tgt: array<f32>;   // live pyramid
@group(0) @binding(2) var<storage, read> tmpl: array<f32>;     // predicted luma pyramid
@group(0) @binding(3) var<storage, read> tdepth: array<f32>;   // predicted inverse depth pyramid
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;
@group(0) @binding(5) var<storage, read_write> mask: array<u32>;

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
        let xi = tdepth[p.dims.w + pi];
        var cls = 0u; // 0 no model, 1 used, 2 rejected, 3 out of view
        if (xi > 0.0) {
            acc[31] += 1.0;
            cls = 3u;
            let xv = vec3f((f32(x) - p.k.z) / p.k.x, (f32(y) - p.k.w) / p.k.y, 1.0) / xi;
            let xl = (t * vec4f(xv, 1.0)).xyz;
            if (xl.z > 1e-6) {
                let uv = vec2f(p.k.x * xl.x / xl.z + p.k.z, p.k.y * xl.y / xl.z + p.k.w);
                if (all(uv >= vec2f(1.0)) && all(uv <= maxc)) {
                    acc[29] += 1.0;
                    let il = sample_tgt(uv);
                    let iv = tmpl[p.dims.w + pi];
                    let r = p.misc.z * il + p.misc.w - iv;
                    if (abs(r) > p.misc.x) {
                        cls = 2u;
                        acc[28] += 1.0;
                    } else {
                        cls = 1u;
                        let a = p.misc.z * image_jacobian(uv, xl);
                        // Chain through R_lv, then the se(3) generators at x_v.
                        let b = vec3f(dot(a, p.t0.xyz), dot(a, p.t1.xyz), dot(a, p.t2.xyz));
                        let rot = cross(xv, b);
                        let j = array<f32, 6>(b.x, b.y, b.z, rot.x, rot.y, rot.z);
                        var idx = 0u;
                        for (var m = 0u; m < 6u; m++) {
                            for (var n = m; n < 6u; n++) {
                                acc[idx] += j[m] * j[n];
                                idx++;
                            }
                            acc[21u + m] += j[m] * r;
                        }
                        acc[27] += r * r;
                        acc[30] += 1.0;
                        acc[32] += il;
                        acc[33] += iv;
                        acc[34] += il * il;
                        acc[35] += il * iv;
                    }
                }
            }
        }
        if (p.misc.y > 0.5) {
            mask[pi] = cls;
        }
    }
    reduce_and_store(acc, lid, wid.x);
}
