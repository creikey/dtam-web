// Average cost C(u, k) and whether any frame observed that voxel.
fn cost_at(k: u32, i: u32, n: u32) -> vec2f {
    let c = (vol_cnt[(k / 4u) * n + i] >> ((k % 4u) * 8u)) & 255u;
    if (c < max(p.min_count, 1u)) {
        return vec2f(0.0, 0.0);
    }
    return vec2f(vol_sum[k * n + i] / f32(c), 1.0);
}
