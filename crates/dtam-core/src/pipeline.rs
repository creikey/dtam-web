//! Per-frame SLAM pipeline. Today it's just the point tracker; camera poses,
//! keyframes and DTAM cost volumes / depth maps will hang off `FrameOutput`.

use crate::{Frame, FrameTracks, Gpu, KltTracker, TrackerParams};

/// Everything the pipeline produced for one frame, for inspection.
#[derive(Clone, Debug, Default)]
pub struct FrameOutput {
    pub tracks: FrameTracks,
}

pub struct SlamPipeline {
    tracker: KltTracker,
}

impl SlamPipeline {
    pub fn new(gpu: Gpu, width: u32, height: u32, params: TrackerParams) -> Self {
        Self {
            tracker: KltTracker::new(gpu, width, height, params),
        }
    }

    pub async fn process(&mut self, frame: &Frame) -> FrameOutput {
        let tracks = self.tracker.process(&frame.luma()).await;
        FrameOutput { tracks }
    }
}
