// Adds one overlapping frame m to the keyframe cost volume (paper eqs. 2-3):
// rho = |I_r(u) - I_m(pi(K T_mr pi^-1(u, d)))|_1 over RGB, for every layer d.
// For x_r = K^-1 u / d:  K x_m * d = (K R K^-1) u + d (K t)  =  a + d b.

struct FrameXf {
    m0: vec4f, // rows of K R_mr K^-1
    m1: vec4f,
    m2: vec4f,
    b: vec4f,  // K t_mr
}

@group(0) @binding(0) var<uniform> p: MapParams;
@group(0) @binding(1) var<uniform> xf: FrameXf;
@group(0) @binding(2) var<storage, read> ref_img: array<u32>;
@group(0) @binding(3) var<storage, read> img: array<u32>;
@group(0) @binding(4) var<storage, read_write> vol_sum: array<f32>;
// Per-voxel sample counts, 4 layers of one pixel packed per u32 (8 bits each).
@group(0) @binding(5) var<storage, read_write> vol_cnt: array<u32>;

fn fetch(x: u32, y: u32) -> vec3f {
    return unpack_rgb(img[y * p.w + x]);
}

fn bilinear(uv: vec2f) -> vec3f {
    let f = floor(uv);
    let t = uv - f;
    let x0 = u32(f.x);
    let y0 = u32(f.y);
    let x1 = min(x0 + 1u, p.w - 1u);
    let y1 = min(y0 + 1u, p.h - 1u);
    let top = mix(fetch(x0, y0), fetch(x1, y0), t.x);
    let bot = mix(fetch(x0, y1), fetch(x1, y1), t.x);
    return mix(top, bot, t.y);
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let n = p.w * p.h;
    let ir = unpack_rgb(ref_img[i]);
    let u = vec3f(f32(id.x), f32(id.y), 1.0);
    let a = vec3f(dot(xf.m0.xyz, u), dot(xf.m1.xyz, u), dot(xf.m2.xyz, u));
    let maxc = vec2f(f32(p.w - 1u), f32(p.h - 1u));
    for (var k = 0u; k < p.layers; k++) {
        let xi = p.xi_min + f32(k) * p.xi_step;
        let q = a + xi * xf.b.xyz;
        if (q.z <= 1e-6) {
            continue;
        }
        let uv = q.xy / q.z;
        if (any(uv < vec2f(0.0)) || any(uv > maxc)) {
            continue;
        }
        let c = bilinear(uv);
        let d = abs(ir - c);
        vol_sum[k * n + i] += d.x + d.y + d.z;
        let ci = (k / 4u) * n + i;
        vol_cnt[ci] += 1u << ((k % 4u) * 8u);
    }
}
