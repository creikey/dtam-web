// Renders one keyframe's inverse depth map as a triangle mesh into the
// virtual camera v (paper §2.3: "project the entire model into a virtual
// camera"). Oblique triangles are culled (paper §2.2.6, "culling oblique
// edges as described in [9]"): a triangle whose normal is nearly
// perpendicular to the keyframe's viewing ray is only bridging a depth
// discontinuity (or a depth outlier), so it is dropped. The normal is the
// exact per-triangle one, from screen-space derivatives of the keyframe-frame
// position. Outputs luma and inverse depth of the prediction.

struct PredictParams {
    t0: vec4f, // T_vr, column-major
    t1: vec4f,
    t2: vec4f,
    t3: vec4f,
    k_r: vec4f, // keyframe fx fy cx cy
    k_v: vec4f, // virtual camera fx fy cx cy
    dims: vec4u, // keyframe w, h, output w, h
    misc: vec4f, // near, far, min |cos| between surface normal and keyframe ray, _
}

@group(0) @binding(0) var<uniform> p: PredictParams;
@group(0) @binding(1) var<storage, read> inv_depth: array<f32>;
@group(0) @binding(2) var<storage, read> rgb: array<u32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) luma: f32,
    @location(1) z: f32,
    @location(2) bad: f32,
    @location(3) pos_r: vec3f, // position in the keyframe camera frame
}

fn point_at(x: i32, y: i32) -> vec3f {
    let w = i32(p.dims.x);
    let h = i32(p.dims.y);
    let cx = clamp(x, 0, w - 1);
    let cy = clamp(y, 0, h - 1);
    let xi = inv_depth[u32(cy * w + cx)];
    return vec3f((f32(cx) - p.k_r.z) / p.k_r.x, (f32(cy) - p.k_r.w) / p.k_r.y, 1.0) / max(xi, 1e-6);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    let w = p.dims.x;
    let x = i32(vi % w);
    let y = i32(vi / w);
    let xi = inv_depth[vi];

    let xr = point_at(x, y);
    var bad = select(0.0, 1.0, xi <= 0.0);

    let t = mat4x4f(p.t0, p.t1, p.t2, p.t3);
    let xv = (t * vec4f(xr, 1.0)).xyz;
    let z = xv.z;
    if (z <= p.misc.x) {
        bad = 1.0;
    }
    let u = p.k_v.x * xv.x / z + p.k_v.z;
    let v = p.k_v.y * xv.y / z + p.k_v.w;
    let ndc = vec2f((u + 0.5) / f32(p.dims.z) * 2.0 - 1.0, 1.0 - (v + 0.5) / f32(p.dims.w) * 2.0);
    let depth = clamp((z - p.misc.x) / (p.misc.y - p.misc.x), 0.0, 1.0);

    let c = rgb[vi];
    let col = vec3f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;

    var out: VOut;
    out.pos = vec4f(ndc * z, depth * z, z);
    out.luma = dot(col, vec3f(0.2126, 0.7152, 0.0722));
    out.z = z;
    out.bad = bad;
    out.pos_r = xr;
    return out;
}

struct FOut {
    @location(0) luma: f32,
    @location(1) inv_depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
    let n = cross(dpdx(in.pos_r), dpdy(in.pos_r));
    let cos_view = abs(dot(normalize(n), normalize(in.pos_r)));
    if (in.bad > 1e-4 || !(cos_view >= p.misc.z)) {
        discard;
    }
    var out: FOut;
    out.luma = in.luma;
    out.inv_depth = 1.0 / in.z;
    return out;
}
