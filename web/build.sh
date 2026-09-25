#!/usr/bin/env bash
# Builds the browser app into web/dist (static files for GitHub Pages).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p dtam-viewer --lib
rm -rf web/dist && mkdir -p web/dist/pkg
wasm-bindgen --target web --no-typescript --out-dir web/dist/pkg \
  target/wasm32-unknown-unknown/release/dtam_viewer.wasm
cp web/index.html web/dtam_preview.jpg web/dist/
# The interactive guide (static files; authoring tools stay out of the site).
mkdir -p web/dist/learn && cp -R web/learn/index.html web/learn/learn.css web/learn/core.js web/learn/chapters web/learn/img web/dist/learn/
# Demo clip: 512x512, short GOP so frame-accurate seeking is fast.
if [ ! -f web/demo.mp4 ]; then
  ffmpeg -v error -y -i test_video.mp4 -vf scale=512:512:flags=area -an \
    -c:v libx264 -preset slow -crf 20 -g 8 -bf 0 -pix_fmt yuv420p -movflags +faststart web/demo.mp4
fi
cp web/demo.mp4 web/dist/
touch web/dist/.nojekyll
du -sh web/dist/* | sort -h
