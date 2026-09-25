//! Small wgpu helpers shared by the GPU stages.
//!
//! Everything here works on native and on wasm/WebGPU: readbacks are async and
//! only block (via `Device::poll`) on native, where there is no event loop to
//! drive the map callbacks.

use wgpu::util::DeviceExt;

/// Number of GPU->CPU round trips so far (profiling).
pub static READBACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A device + queue pair. Cheap to clone; the viewer shares its render device
/// with the SLAM pipeline so everything runs on one GPU context.
#[derive(Clone)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
    }

    /// Creates a standalone compute-only context (tests, headless tools).
    pub async fn headless() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("request_device failed: {e}"))?;
        Ok(Self { device, queue })
    }

    pub(crate) fn compute_pipeline(&self, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
        self.compute_pipeline_entry(label, wgsl, "cs_main")
    }

    pub(crate) fn compute_pipeline_entry(&self, label: &str, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            // Every kernel writes its workgroup memory before reading it, so
            // skip the zero-fill: in wgpu's native backends it is done
            // serially and costs ~0.2 ms per dispatch of a reduction kernel
            // (browsers always zero-fill, efficiently).
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        })
    }

    pub(crate) fn storage(&self, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub(crate) fn readback(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).next_multiple_of(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn storage_init(&self, label: &str, data: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | extra,
        })
    }

    pub(crate) fn bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Maps the first `sizes[i]` bytes of each buffer and returns copies.
    pub(crate) async fn read_buffers(&self, bufs: &[(&wgpu::Buffer, u64)]) -> Vec<Vec<u8>> {
        READBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut receivers = Vec::new();
        for (buf, size) in bufs {
            let (tx, rx) = futures_channel::oneshot::channel();
            buf.slice(..(*size).max(4).next_multiple_of(4))
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            receivers.push(rx);
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        let mut out = Vec::new();
        for ((buf, size), rx) in bufs.iter().zip(receivers) {
            rx.await
                .expect("map callback dropped")
                .expect("buffer map failed");
            let bytes = {
                let view = buf
                    .slice(..(*size).max(4).next_multiple_of(4))
                    .get_mapped_range()
                    .expect("mapped range");
                view[..*size as usize].to_vec()
            };
            buf.unmap();
            out.push(bytes);
        }
        out
    }
}

pub(crate) fn dispatch_2d(pass: &mut wgpu::ComputePass, w: u32, h: u32) {
    pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
}
