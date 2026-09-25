// Types shared by the dense alignment passes and the on-GPU solver.

struct TrackParams {
    t0: vec4f, // transform, column-major
    t1: vec4f,
    t2: vec4f,
    t3: vec4f,
    k: vec4f,     // fx fy cx cy at this level
    dims: vec4u,  // w, h, target offset, template offset
    misc: vec4f,  // outlier threshold, mode (0 solve, 1 mask, 2 fixed t), gain, bias (mode 2)
}

// Solver state kept on the GPU across all Gauss-Newton iterations of a frame,
// so the CPU reads back once per frame instead of once per iteration.
struct GnState {
    cand: mat4x4f,          // transform the next residual pass evaluates
    pose: mat4x4f,          // last accepted transform
    sums: array<f32, 36>,   // accumulators at the accepted transform
    best: f32,              // robust cost at the accepted transform
    damping: f32,
    gain: f32,
    bias: f32,
    done: u32,              // this level has converged or failed
    has_acc: u32,           // a transform was accepted on this level
    iters: u32,
    _pad: u32,
}

