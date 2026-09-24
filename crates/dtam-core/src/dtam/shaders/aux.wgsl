// Point-wise search for a = argmin (d - a)^2 / (2 theta) + lambda C(u, a)
// (paper eqs. 13-14), restricted to the feasible band |a - d| <= r with
// r = sqrt(2 theta lambda (C_max - C_min)) (§2.2.4), then one Newton step on
// the sampled energy for sub-sample accuracy (eq. 18).

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<storage, read> vol_sum: array<f32>;
@group(0) @binding(2) var<storage, read> vol_cnt: array<u32>;
@group(0) @binding(3) var<storage, read> stats: array<vec4f>;
@group(0) @binding(4) var<storage, read> d: array<f32>;
@group(0) @binding(5) var<storage, read_write> a: array<f32>;

fn energy(k: u32, i: u32, n: u32, du: f32, cmax: f32) -> f32 {
    let xi = p.xi_min + f32(k) * p.xi_step;
    let c = cost_at(k, i, n);
    let data = select(cmax, c.x, c.y > 0.0);
    let diff = du - xi;
    return diff * diff / (2.0 * p.theta) + p.lambda * data;
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let n = p.w * p.h;
    let s = stats[i];
    let du = d[i];
    let last = f32(p.layers - 1u);
    let r = sqrt(2.0 * p.theta * p.lambda * max(s.y - s.x, 0.0));
    let k0 = u32(clamp(floor((du - r - p.xi_min) / p.xi_step), 0.0, last));
    let k1 = u32(clamp(ceil((du + r - p.xi_min) / p.xi_step), 0.0, last));

    var best = k0;
    var e_best = energy(k0, i, n, du, s.y);
    for (var k = k0 + 1u; k <= k1; k++) {
        let e = energy(k, i, n, du, s.y);
        if (e < e_best) {
            e_best = e;
            best = k;
        }
    }

    var xi = p.xi_min + f32(best) * p.xi_step;
    if (best > 0u && best + 1u < p.layers) {
        let em = energy(best - 1u, i, n, du, s.y);
        let ep = energy(best + 1u, i, n, du, s.y);
        let grad = (ep - em) / (2.0 * p.xi_step);
        let hess = (ep - 2.0 * e_best + em) / (p.xi_step * p.xi_step);
        if (hess > 0.0) {
            xi -= clamp(grad / hess, -p.xi_step, p.xi_step);
        }
    }
    a[i] = xi;
}
