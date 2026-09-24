// Unpacks 8-bit luma (4 pixels per u32) into pyramid level 0 as f32 in [0, 1].

struct Params {
    dst_off: u32,
    w: u32,
    h: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> pyr: array<f32>;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let i = id.y * p.w + id.x;
    let v = (src[i / 4u] >> ((i % 4u) * 8u)) & 0xffu;
    pyr[p.dst_off + i] = f32(v) / 255.0;
}
