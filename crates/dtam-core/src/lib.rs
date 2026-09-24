//! DTAM SLAM core: GPU (wgpu) processing stages, no windowing or I/O.
//!
//! Runs on native (Metal/Vulkan/DX12) and is written to also run on
//! wasm32 + browser WebGPU: all GPU readbacks are async.

pub mod calib;
pub mod frame;
pub mod gpu;
pub mod pipeline;
pub mod tracker;

pub use frame::Frame;
pub use gpu::Gpu;
pub use pipeline::{FrameOutput, SlamPipeline};
pub use tracker::{FrameTracks, KltTracker, TrackedPoint, TrackerParams};
