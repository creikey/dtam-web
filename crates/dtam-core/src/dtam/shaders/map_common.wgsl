// Shared by the mapping passes (prepended to each shader).

struct MapParams {
    w: u32,
    h: u32,
    layers: u32,
    min_count: u32, // voxels seen by fewer frames count as unobserved
    xi_min: f32,   // inverse depth of layer 0
    xi_step: f32,  // inverse depth spacing between layers
    theta: f32,    // coupling strength (paper eq. 7)
    lambda: f32,   // data term weight
    eps: f32,      // Huber epsilon
    sigma_q: f32,  // dual step
    sigma_d: f32,  // primal step
    alpha: f32,    // g(u) = exp(-alpha |grad I|^beta)
    beta: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}

fn unpack_rgb(p: u32) -> vec3f {
    return vec3f(f32(p & 255u), f32((p >> 8u) & 255u), f32((p >> 16u) & 255u)) / 255.0;
}

fn luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}
