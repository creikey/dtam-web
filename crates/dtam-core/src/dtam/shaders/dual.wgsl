// Dual ascent: q = Proj_{|q|<=1}((q + sigma_q g grad d) / (1 + sigma_q eps)).
// grad uses forward differences with Neumann boundary.

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> d: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<vec2f>;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    var grad = vec2f(0.0);
    if (id.x + 1u < p.w) {
        grad.x = d[i + 1u] - d[i];
    }
    if (id.y + 1u < p.h) {
        grad.y = d[i + p.w] - d[i];
    }
    let qn = (q[i] + p.sigma_q * g[i] * grad) / (1.0 + p.sigma_q * p.eps);
    q[i] = qn / max(1.0, length(qn));
}
