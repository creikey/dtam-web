// 2x decimation with a separable [1 3 3 1]/8 filter (pixel-centered).

struct Params {
    src_off: u32,
    src_w: u32,
    src_h: u32,
    dst_off: u32,
    dst_w: u32,
    dst_h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> pyr: array<f32>;

fn px(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(p.src_w) - 1));
    let cy = u32(clamp(y, 0, i32(p.src_h) - 1));
    return pyr[p.src_off + cy * p.src_w + cx];
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.dst_w || id.y >= p.dst_h) {
        return;
    }
    let k = array<f32, 4>(1.0, 3.0, 3.0, 1.0);
    let x0 = i32(id.x) * 2 - 1;
    let y0 = i32(id.y) * 2 - 1;
    var acc = 0.0;
    for (var j = 0; j < 4; j++) {
        var row = 0.0;
        for (var i = 0; i < 4; i++) {
            row += k[i] * px(x0 + i, y0 + j);
        }
        acc += k[j] * row;
    }
    pyr[p.dst_off + id.y * p.dst_w + id.x] = acc / 64.0;
}
