#!/usr/bin/env python3
"""Convert an iPhone HLG/HDR video to a standardized SDR RGB mp4 for CV.

Output: 1024x1024 center crop, SDR BT.709, browser-compatible H.264 High
profile yuv420p (decodes to 8-bit 3-channel RGB), no audio.

Usage: scripts/convert_video.py input.MOV output.mp4
"""
import subprocess
import sys

import numpy as np

SIZE = 1024
CRF = "18"

# HLG constants (ITU-R BT.2100)
A, B, C = 0.17883277, 0.28466892, 0.55991073
PEAK_NITS = 1000.0
REF_WHITE_NITS = 203.0  # HLG 75% signal = SDR diffuse white

BT2020_TO_BT709 = np.array([
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
], dtype=np.float32)
BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
BT709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def hlg_to_srgb(frame_u16: np.ndarray) -> np.ndarray:
    e = frame_u16.astype(np.float32) / 65535.0

    # Inverse OETF -> scene linear
    lin = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - C) / A) + B) / 12.0)

    # OOTF -> display light (nits), then relative to SDR white
    ys = lin @ BT2020_LUMA
    gain = PEAK_NITS * np.power(np.maximum(ys, 1e-6), 0.2) / REF_WHITE_NITS
    rgb = lin * gain[..., None]

    # BT.2020 -> BT.709 primaries
    rgb = np.maximum(rgb @ BT2020_TO_BT709.T, 0.0)

    # Extended Reinhard on luminance: compresses the full HDR range, peak -> 1.0
    y = rgb @ BT709_LUMA
    peak = PEAK_NITS / REF_WHITE_NITS
    y_mapped = y * (1 + y / (peak * peak)) / (1 + y)
    rgb *= (y_mapped / np.maximum(y, 1e-6))[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)

    # sRGB OETF
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def main(src: str, dst: str) -> None:
    decode = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", src, "-map", "0:v:0",
        "-vf", f"crop=ih:ih,scale={SIZE}:{SIZE}:flags=lanczos"
               ":in_color_matrix=bt2020:in_range=tv:out_range=pc,format=rgb48le",
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "-",
    ], stdout=subprocess.PIPE)
    rate = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src,
    ], text=True).strip().rstrip(",")
    encode = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", rate, "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
        "-c:v", "libx264", "-preset", "slow", "-crf", CRF, "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", dst,
    ], stdin=subprocess.PIPE)

    frame_bytes = SIZE * SIZE * 3 * 2
    n = 0
    while chunk := decode.stdout.read(frame_bytes):
        if len(chunk) < frame_bytes:
            break
        frame = np.frombuffer(chunk, dtype="<u2").reshape(SIZE, SIZE, 3)
        encode.stdin.write(hlg_to_srgb(frame).tobytes())
        n += 1
    encode.stdin.close()
    if decode.wait() or encode.wait():
        sys.exit("ffmpeg failed")
    print(f"wrote {n} frames to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
