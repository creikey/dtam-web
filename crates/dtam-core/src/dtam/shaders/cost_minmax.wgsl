// Per pixel: C_min, C_max, arg min (paper §2.2.4), and d0 = a0 = arg min C.

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<storage, read> vol_sum: array<f32>;
@group(0) @binding(2) var<storage, read> vol_cnt: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<vec4f>; // cmin, cmax, xi at cmin, #valid layers
@group(0) @binding(4) var<storage, read_write> d: array<f32>;
@group(0) @binding(5) var<storage, read_write> a: array<f32>;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let n = p.w * p.h;
    var cmin = 1e30;
    var cmax = 0.0;
    var kmin = (p.layers - 1u) / 2u;
    var valid = 0u;
    for (var k = 0u; k < p.layers; k++) {
        let c = cost_at(k, i, n);
        if (c.y == 0.0) {
            continue;
        }
        valid += 1u;
        if (c.x < cmin) {
            cmin = c.x;
            kmin = k;
        }
        cmax = max(cmax, c.x);
    }
    if (valid == 0u) {
        cmin = 0.0;
    }
    let xi = p.xi_min + f32(kmin) * p.xi_step;
    stats[i] = vec4f(cmin, cmax, xi, f32(valid));
    d[i] = xi;
    a[i] = xi;
}
