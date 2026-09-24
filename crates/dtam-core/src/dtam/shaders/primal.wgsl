// Primal descent: d = (d + sigma_d (div(g q) + a / theta)) / (1 + sigma_d / theta).
// div is the negative adjoint of the forward-difference gradient.

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<vec2f>;
@group(0) @binding(3) var<storage, read> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    var div = 0.0;
    if (id.x + 1u < p.w) {
        div += g[i] * q[i].x;
    }
    if (id.x > 0u) {
        div -= g[i - 1u] * q[i - 1u].x;
    }
    if (id.y + 1u < p.h) {
        div += g[i] * q[i].y;
    }
    if (id.y > 0u) {
        div -= g[i - p.w] * q[i - p.w].y;
    }
    d[i] = (d[i] + p.sigma_d * (div + a[i] / p.theta)) / (1.0 + p.sigma_d / p.theta);
}
