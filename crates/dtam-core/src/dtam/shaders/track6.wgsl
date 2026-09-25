// 6DOF forward-compositional alignment of the live image to the model
// prediction (paper eqs. 19-21): f_u = I_l(pi(K T_lv exp(psi) pi^-1(u, xi_v(u)))) - I_v(u),
// with a global gain/bias on I_l (a I_l + b) for robustness to exposure
// changes (the affine invariance NCC gives, paper §3.2 / [5]).
// Pixels with |f_u| above a threshold are rejected (paper §2.3.2).

@group(0) @binding(0) var<uniform> p: TrackParams;
@group(0) @binding(1) var<storage, read> tgt: array<f32>;   // live pyramid
@group(0) @binding(2) var<storage, read> tmpl: array<f32>;     // predicted luma pyramid
@group(0) @binding(3) var<storage, read> tdepth: array<f32>;   // predicted inverse depth pyramid
@group(0) @binding(4) var<storage, read_write> partials: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> mask: array<u32>;
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
    // Mode 0: evaluate the solver's candidate (nothing left to do once the
    // level is done); 1: the accepted transform, writing the mask; 2: the
    // transform and gain/bias given in the uniform.
    let mode = u32(p.misc.y + 0.5);
    let npx = select(w * h, 0u, mode == 0u && st.done != 0u);
    var t = mat4x4f(p.t0, p.t1, p.t2, p.t3);
    var gain = p.misc.z;
    var bias = p.misc.w;
    if (mode == 0u) {
        t = st.cand;
    }
    if (mode == 1u) {
        t = st.pose;
    }
    if (mode != 2u) {
        gain = st.gain;
        bias = st.bias;
    }
    let maxc = vec2f(f32(w) - 2.0, f32(h) - 2.0);
    for (var pi = wid.x * WG + lid; pi < npx; pi += nwg.x * WG) {
        let x = pi % w;
        let y = pi / w;
        let xi = tdepth[p.dims.w + pi];
        var cls = 0u; // 0 no model, 1 used, 2 rejected, 3 out of view
        if (xi > 0.0) {
            acc[7].w += 1.0;
            cls = 3u;
            let xv = vec3f((f32(x) - p.k.z) / p.k.x, (f32(y) - p.k.w) / p.k.y, 1.0) / xi;
            let xl = (t * vec4f(xv, 1.0)).xyz;
            if (xl.z > 1e-6) {
                let uv = vec2f(p.k.x * xl.x / xl.z + p.k.z, p.k.y * xl.y / xl.z + p.k.w);
                if (all(uv >= vec2f(1.0)) && all(uv <= maxc)) {
                    acc[7].y += 1.0;
                    let il = sample_tgt(uv);
                    let iv = tmpl[p.dims.w + pi];
                    let r = gain * il + bias - iv;
                    if (abs(r) > p.misc.x) {
                        cls = 2u;
                        acc[7].x += 1.0;
                    } else {
                        cls = 1u;
                        let a = gain * image_jacobian(uv, xl);
                        // Chain through R_lv, then the se(3) generators at x_v.
                        let b = vec3f(dot(a, t[0].xyz), dot(a, t[1].xyz), dot(a, t[2].xyz));
                        let rot = cross(xv, b);
                        // J = (b, x_v x b); upper triangle of J^T J, then J^T r.
                        let j0 = b.x;
                        let j1 = b.y;
                        let j2 = b.z;
                        let j3 = rot.x;
                        let j4 = rot.y;
                        let j5 = rot.z;
                        acc[0] += j0 * vec4f(j0, j1, j2, j3);
                        acc[1] += vec4f(j0 * j4, j0 * j5, j1 * j1, j1 * j2);
                        acc[2] += vec4f(j1 * j3, j1 * j4, j1 * j5, j2 * j2);
                        acc[3] += vec4f(j2 * j3, j2 * j4, j2 * j5, j3 * j3);
                        acc[4] += vec4f(j3 * j4, j3 * j5, j4 * j4, j4 * j5);
                        acc[5] += vec4f(j5 * j5, j0 * r, j1 * r, j2 * r);
                        acc[6] += vec4f(j3 * r, j4 * r, j5 * r, r * r);
                        acc[7].z += 1.0;
                        acc[8] += vec4f(il, iv, il * il, il * iv);
                    }
                }
            }
        }
        if (mode == 1u) {
            mask[pi] = cls;
        }
    }
    reduce_and_store(acc, lid, wid.x);
}
