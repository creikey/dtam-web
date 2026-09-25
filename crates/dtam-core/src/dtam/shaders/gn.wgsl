// On-GPU Gauss-Newton / Levenberg-Marquardt steps for the dense alignment
// passes (track6 / track_rot): reduce the per-workgroup partial sums, solve the
// normal equations and update the transform in `st`. The logic mirrors the
// per-iteration CPU loop it replaces, so a whole frame's coarse-to-fine
// alignment is one submit and one readback.

@group(0) @binding(0) var<uniform> p: TrackParams;
@group(0) @binding(4) var<storage, read> partials: array<f32>;
@group(0) @binding(6) var<storage, read_write> st: GnState;

const NACC: u32 = 36u;
const NWG: u32 = 128u;

var<workgroup> sums: array<f32, NACC>;

// Sums the NWG partial rows; one thread per accumulator.
fn reduce(lid: u32) {
    if (lid < NACC) {
        var s = 0.0;
        for (var w = 0u; w < NWG; w++) {
            s += partials[w * NACC + lid];
        }
        sums[lid] = s;
    }
    workgroupBarrier();
}

// Solves H x = b (n <= 6, H symmetric positive definite) with Jacobi scaling
// and Cholesky; returns false if H is not positive definite.
var<private> hm: array<f32, 36>;
var<private> bv: array<f32, 6>;
var<private> xv: array<f32, 6>;

fn solve(n: u32) -> bool {
    var sc: array<f32, 6>;
    for (var i = 0u; i < n; i++) {
        let d = hm[i * 6u + i];
        if (!(d > 0.0)) {
            return false;
        }
        sc[i] = inverseSqrt(d);
    }
    for (var i = 0u; i < n; i++) {
        for (var j = 0u; j < n; j++) {
            hm[i * 6u + j] *= sc[i] * sc[j];
        }
        bv[i] *= sc[i];
    }
    // In-place Cholesky: lower triangle holds L.
    for (var j = 0u; j < n; j++) {
        var d = hm[j * 6u + j];
        for (var k = 0u; k < j; k++) {
            d -= hm[j * 6u + k] * hm[j * 6u + k];
        }
        if (!(d > 1e-12)) {
            return false;
        }
        let l = sqrt(d);
        hm[j * 6u + j] = l;
        for (var i = j + 1u; i < n; i++) {
            var s = hm[i * 6u + j];
            for (var k = 0u; k < j; k++) {
                s -= hm[i * 6u + k] * hm[j * 6u + k];
            }
            hm[i * 6u + j] = s / l;
        }
    }
    var y: array<f32, 6>;
    for (var i = 0u; i < n; i++) {
        var s = bv[i];
        for (var k = 0u; k < i; k++) {
            s -= hm[i * 6u + k] * y[k];
        }
        y[i] = s / hm[i * 6u + i];
    }
    for (var ii = 0u; ii < n; ii++) {
        let i = n - 1u - ii;
        var s = y[i];
        for (var k = i + 1u; k < n; k++) {
            s -= hm[k * 6u + i] * xv[k];
        }
        xv[i] = s / hm[i * 6u + i];
    }
    for (var i = 0u; i < n; i++) {
        xv[i] *= sc[i];
    }
    return true;
}

fn skew(w: vec3f) -> mat3x3f {
    // Column-major: columns of [w]x.
    return mat3x3f(vec3f(0.0, w.z, -w.y), vec3f(-w.z, 0.0, w.x), vec3f(w.y, -w.x, 0.0));
}

// Gram-Schmidt re-orthonormalisation of a rotation.
fn orthonormalize(r: mat3x3f) -> mat3x3f {
    let c0 = normalize(r[0]);
    let c1 = normalize(r[1] - dot(r[1], c0) * c0);
    return mat3x3f(c0, c1, cross(c0, c1));
}

// SE(3) exponential of the twist (v, w), translation part first.
fn se3_exp(v: vec3f, w: vec3f) -> mat4x4f {
    let th = length(w);
    let wx = skew(w);
    let wx2 = wx * wx;
    var a = 1.0 - th * th / 6.0;
    var b = 0.5 - th * th / 24.0;
    var c = 1.0 / 6.0 - th * th / 120.0;
    if (th > 1e-4) {
        a = sin(th) / th;
        b = (1.0 - cos(th)) / (th * th);
        c = (th - sin(th)) / (th * th * th);
    }
    let id = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    let r = orthonormalize(id + wx * a + wx2 * b);
    let t = (id + wx * b + wx2 * c) * v;
    return mat4x4f(vec4f(r[0], 0.0), vec4f(r[1], 0.0), vec4f(r[2], 0.0), vec4f(t, 1.0));
}

fn rot3(m: mat4x4f) -> mat3x3f {
    return mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
}

// Start of a pyramid level: continue from the accepted transform.
@compute @workgroup_size(1)
fn begin() {
    st.cand = st.pose;
    st.has_acc = 0u;
    st.damping = 1e-4;
    st.done = 0u;
}

// One LM-safeguarded Gauss-Newton step of the 6DOF alignment: a candidate is
// accepted only if the robust (truncated-quadratic) cost drops; otherwise
// revert to the accepted transform and damp harder.
@compute @workgroup_size(64)
fn step6(@builtin(local_invocation_index) lid: u32) {
    reduce(lid);
    if (lid != 0u || st.done != 0u) {
        return;
    }
    st.iters += 1u;
    if (sums[29] < 100.0 || sums[30] < 100.0) {
        st.done = 1u;
        return;
    }
    let t2 = p.misc.x * p.misc.x;
    let cost = (sums[27] + sums[28] * t2) / sums[29];
    if (st.has_acc != 0u && cost > st.best) {
        st.damping *= 10.0;
        if (st.damping > 1e4) {
            st.done = 1u;
            return;
        }
    } else {
        st.damping = max(st.damping / 3.0, 1e-7);
        // Closed-form gain/bias for the accepted warp.
        let n = sums[30];
        let det = n * sums[34] - sums[32] * sums[32];
        if (n > 100.0 && abs(det) > 1e-9) {
            let a = clamp((n * sums[35] - sums[32] * sums[33]) / det, 0.5, 2.0);
            st.bias = (sums[33] - a * sums[32]) / n;
            st.gain = a;
        }
        st.has_acc = 1u;
        st.best = cost;
        st.pose = st.cand;
        for (var k = 0u; k < NACC; k++) {
            st.sums[k] = sums[k];
        }
    }
    // Step from the accepted transform using its normal equations.
    var idx = 0u;
    for (var m = 0u; m < 6u; m++) {
        for (var n = m; n < 6u; n++) {
            hm[m * 6u + n] = st.sums[idx];
            hm[n * 6u + m] = st.sums[idx];
            idx++;
        }
        bv[m] = -st.sums[21u + m];
    }
    for (var i = 0u; i < 6u; i++) {
        hm[i * 6u + i] *= 1.0 + st.damping;
    }
    if (!solve(6u)) {
        st.done = 1u;
        return;
    }
    let v = vec3f(xv[0], xv[1], xv[2]);
    let w = vec3f(xv[3], xv[4], xv[5]);
    st.cand = st.pose * se3_exp(v, w);
    if (length(v) + length(w) < 1e-6) {
        st.done = 1u;
    }
}

// One Gauss-Newton step of the inter-frame rotation: R <- exp(w) R.
@compute @workgroup_size(64)
fn step_rot(@builtin(local_invocation_index) lid: u32) {
    reduce(lid);
    if (lid != 0u || st.done != 0u) {
        return;
    }
    if (sums[30] < 50.0) {
        st.done = 1u;
        return;
    }
    hm[0] = sums[0]; hm[1] = sums[1]; hm[2] = sums[2];
    hm[6] = sums[1]; hm[7] = sums[3]; hm[8] = sums[4];
    hm[12] = sums[2]; hm[13] = sums[4]; hm[14] = sums[5];
    bv[0] = -sums[21];
    bv[1] = -sums[22];
    bv[2] = -sums[23];
    if (!solve(3u)) {
        st.done = 1u;
        return;
    }
    let w = vec3f(xv[0], xv[1], xv[2]);
    let r = orthonormalize(rot3(se3_exp(vec3f(0.0), w)) * rot3(st.cand));
    st.cand = mat4x4f(vec4f(r[0], 0.0), vec4f(r[1], 0.0), vec4f(r[2], 0.0), vec4f(0.0, 0.0, 0.0, 1.0));
    st.pose = st.cand;
    if (length(w) < 1e-6) {
        st.done = 1u;
    }
}
