// Pyramidal Lucas-Kanade, one invocation per point. Each point is tracked
// prev -> cur, then back cur -> prev for a forward-backward consistency check.
// Both frames' pyramids live in one buffer, in two "slots".

const MAX_LEVELS: u32 = 6u;
const EPS: f32 = 0.01;

struct Level {
    off: u32, // offset within a slot
    w: u32,
    h: u32,
    _pad: u32,
}

struct Params {
    levels: array<Level, MAX_LEVELS>,
    slot_size: u32,
    num_levels: u32,
    prev_slot: u32,
    cur_slot: u32,
    num_points: u32,
    radius: i32,
    max_iters: u32,
    _pad: u32,
}

struct PointIn {
    pos: vec2f,   // position in prev frame
    guess: vec2f, // initial guess in cur frame
}

struct PointOut {
    pos: vec2f,  // tracked position in cur frame
    back: vec2f, // position after tracking back into prev frame
    residual: f32,
    status: u32, // 1 = ok
    _pad: vec2u,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pyr: array<f32>;
@group(0) @binding(2) var<storage, read> pts_in: array<PointIn>;
@group(0) @binding(3) var<storage, read_write> pts_out: array<PointOut>;

fn sample(slot: u32, l: u32, q: vec2f) -> f32 {
    let lv = p.levels[l];
    let base = slot * p.slot_size + lv.off;
    let maxc = vec2f(f32(lv.w - 1u), f32(lv.h - 1u));
    let c = clamp(q, vec2f(0.0), maxc);
    let i0 = vec2u(floor(c));
    let i1 = min(i0 + 1u, vec2u(lv.w - 1u, lv.h - 1u));
    let f = c - floor(c);
    let a = pyr[base + i0.y * lv.w + i0.x];
    let b = pyr[base + i0.y * lv.w + i1.x];
    let cc = pyr[base + i1.y * lv.w + i0.x];
    let d = pyr[base + i1.y * lv.w + i1.x];
    return mix(mix(a, b, f.x), mix(cc, d, f.x), f.y);
}

// Level-0 pixel coords -> level-l pixel coords (pixel-centered decimation).
fn to_level(q: vec2f, l: u32) -> vec2f {
    let s = 1.0 / f32(1u << l);
    return (q + 0.5) * s - 0.5;
}

struct TrackResult {
    pos: vec2f,
    residual: f32,
    ok: bool,
}

fn track(from_slot: u32, to_slot: u32, p0: vec2f, guess: vec2f) -> TrackResult {
    let top = p.num_levels - 1u;
    var d = (guess - p0) / f32(1u << top);
    var ok = true;
    let r = p.radius;

    for (var li = i32(top); li >= 0; li--) {
        let l = u32(li);
        let pl = to_level(p0, l);

        // Structure tensor of the template window.
        var gxx = 0.0;
        var gxy = 0.0;
        var gyy = 0.0;
        for (var y = -r; y <= r; y++) {
            for (var x = -r; x <= r; x++) {
                let q = pl + vec2f(f32(x), f32(y));
                let ix = 0.5 * (sample(from_slot, l, q + vec2f(1.0, 0.0)) - sample(from_slot, l, q - vec2f(1.0, 0.0)));
                let iy = 0.5 * (sample(from_slot, l, q + vec2f(0.0, 1.0)) - sample(from_slot, l, q - vec2f(0.0, 1.0)));
                gxx += ix * ix;
                gxy += ix * iy;
                gyy += iy * iy;
            }
        }
        let det = gxx * gyy - gxy * gxy;
        let n = f32((2 * r + 1) * (2 * r + 1));
        let tr = gxx + gyy;
        let min_eig = 0.5 * (tr - sqrt(max((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy, 0.0))) / n;
        if (min_eig < 1e-7 || abs(det) < 1e-12) {
            ok = false;
            break;
        }
        let inv = vec3f(gyy, -gxy, gxx) / det; // [a b; b c]

        for (var it = 0u; it < p.max_iters; it++) {
            var bx = 0.0;
            var by = 0.0;
            for (var y = -r; y <= r; y++) {
                for (var x = -r; x <= r; x++) {
                    let q = pl + vec2f(f32(x), f32(y));
                    let ix = 0.5 * (sample(from_slot, l, q + vec2f(1.0, 0.0)) - sample(from_slot, l, q - vec2f(1.0, 0.0)));
                    let iy = 0.5 * (sample(from_slot, l, q + vec2f(0.0, 1.0)) - sample(from_slot, l, q - vec2f(0.0, 1.0)));
                    let diff = sample(from_slot, l, q) - sample(to_slot, l, q + d);
                    bx += diff * ix;
                    by += diff * iy;
                }
            }
            let delta = vec2f(inv.x * bx + inv.y * by, inv.y * bx + inv.z * by);
            d += delta;
            if (dot(delta, delta) < EPS * EPS) {
                break;
            }
        }
        if (l > 0u) {
            d *= 2.0;
        }
    }

    let pos = p0 + d;
    var residual = 0.0;
    if (ok) {
        for (var y = -r; y <= r; y++) {
            for (var x = -r; x <= r; x++) {
                let o = vec2f(f32(x), f32(y));
                residual += abs(sample(from_slot, 0u, p0 + o) - sample(to_slot, 0u, pos + o));
            }
        }
        residual /= f32((2 * r + 1) * (2 * r + 1));
        let lv = p.levels[0];
        if (pos.x < 0.0 || pos.y < 0.0 || pos.x > f32(lv.w - 1u) || pos.y > f32(lv.h - 1u)) {
            ok = false;
        }
    }
    return TrackResult(pos, residual, ok);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.num_points) {
        return;
    }
    let pin = pts_in[i];
    let fwd = track(p.prev_slot, p.cur_slot, pin.pos, pin.guess);
    var out: PointOut;
    out.pos = fwd.pos;
    out.back = vec2f(-1e9);
    out.residual = fwd.residual;
    out.status = 0u;
    if (fwd.ok) {
        let bwd = track(p.cur_slot, p.prev_slot, fwd.pos, pin.pos);
        if (bwd.ok) {
            out.back = bwd.pos;
            out.status = 1u;
        }
    }
    pts_out[i] = out;
}
