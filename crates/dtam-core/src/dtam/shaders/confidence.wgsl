// Per-pixel confidence of the final inverse depth: how much better the
// photometric cost at the solution is than the average over all depth
// samples, (C_mean - C(xi)) / C_mean, times the fraction of layers observed.
// Textureless / self-similar pixels match equally well at every depth -> ~0.

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<storage, read> vol_sum: array<f32>;
@group(0) @binding(2) var<storage, read> vol_cnt: array<u32>;
@group(0) @binding(3) var<storage, read> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> conf: array<vec2f>; // confidence, C(xi)

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let n = p.w * p.h;
    var sum = 0.0;
    var valid = 0.0;
    for (var k = 0u; k < p.layers; k++) {
        let c = cost_at(k, i, n);
        sum += c.x;
        valid += c.y;
    }
    if (valid < 2.0) {
        conf[i] = vec2f(0.0, 0.0);
        return;
    }
    let mean = sum / valid;
    // Cost at the solution, linearly interpolated between layers.
    let t = clamp((d[i] - p.xi_min) / p.xi_step, 0.0, f32(p.layers - 1u));
    let k0 = u32(floor(t));
    let k1 = min(k0 + 1u, p.layers - 1u);
    let c0 = cost_at(k0, i, n);
    let c1 = cost_at(k1, i, n);
    var c = mean;
    if (c0.y > 0.0 && c1.y > 0.0) {
        c = mix(c0.x, c1.x, t - f32(k0));
    } else if (c0.y > 0.0) {
        c = c0.x;
    } else if (c1.y > 0.0) {
        c = c1.x;
    }
    let score = clamp((mean - c) / max(mean, 1e-6), 0.0, 1.0) * valid / f32(p.layers);
    conf[i] = vec2f(score, c);
}
