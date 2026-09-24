// Shi-Tomasi (min eigenvalue) corner response on pyramid level 0, reduced to
// the single strongest pixel per 16x16 tile.

struct Params {
    off: u32,
    w: u32,
    h: u32,
    border: u32,
}

struct Candidate {
    xy: u32, // x | (y << 16)
    score: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pyr: array<f32>;
@group(0) @binding(2) var<storage, read_write> cands: array<Candidate>;

const RADIUS: i32 = 2;

var<workgroup> best_score: array<f32, 256>;
var<workgroup> best_xy: array<u32, 256>;

fn px(x: i32, y: i32) -> f32 {
    return pyr[p.off + u32(y) * p.w + u32(x)];
}

fn response(x: i32, y: i32) -> f32 {
    var sxx = 0.0;
    var sxy = 0.0;
    var syy = 0.0;
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let u = x + dx;
            let v = y + dy;
            let gx = 0.5 * (px(u + 1, v) - px(u - 1, v));
            let gy = 0.5 * (px(u, v + 1) - px(u, v - 1));
            sxx += gx * gx;
            sxy += gx * gy;
            syy += gy * gy;
        }
    }
    let n = f32((2 * RADIUS + 1) * (2 * RADIUS + 1));
    sxx /= n;
    sxy /= n;
    syy /= n;
    let d = sxx - syy;
    return 0.5 * (sxx + syy - sqrt(d * d + 4.0 * sxy * sxy));
}

@compute @workgroup_size(16, 16)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
) {
    var score = 0.0;
    let b = max(p.border, u32(RADIUS + 1));
    if (gid.x >= b && gid.y >= b && gid.x + b < p.w && gid.y + b < p.h) {
        score = response(i32(gid.x), i32(gid.y));
    }
    best_score[lid] = score;
    best_xy[lid] = gid.x | (gid.y << 16u);
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s && best_score[lid + s] > best_score[lid]) {
            best_score[lid] = best_score[lid + s];
            best_xy[lid] = best_xy[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        cands[wid.y * nwg.x + wid.x] = Candidate(best_xy[0], best_score[0]);
    }
}
