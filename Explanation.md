(Not written by AI)

Each frame is 1024x1024 of RGB pixels with colors tuned for correct human perception.

A 'luma' or grayscale copy of the frame is created according to human perceptual brightness weights of the RGB channels of the image for tracking, which looks the most like a grayscale version of the image to a human.

Each frame has a number of tracked points. A tracked point has a stable identity across frames and can die if it can't be found in a new frame.

At each frame:
The frame is converted to luma, copied on the heap then sent to the gpu
All current points are also written to the gpu
Each pyramid level has an offset into its buffer, a width and a height
Those pyramid level offsets are also sent to the gpu

Then, the actual compute pipelines run:
- unpack converts luma to float
  - the shader reads from array<u32> and unpacks the four pixels in the u32 into four separate float32 samples on the output storage buffer because wgsl doesn't have u8 types.
- downsample creates pyramid levels for each desired res
  - the downsample passes are 16x16 workgroup size, dispatched for the resolution of the lower res (each thread is a lower res pixel)
  - the 16x16 in this case is arbitrary and unrelated to any resolutions
  - each lower res pixel is a pixel centered in a 4x4 block in the higher res pixel.  
  - this specific filter doesn't matter that much, it's just downressing and eliminating details
- klt_pipe attempts to track existing points through time
  - a luminosity patch of radius 7 (so 15x15) is tracked over time, starting with a guess
  - in the guess patch, using each pixel's calculated gradient, brightness differences from what the track should look like are used in a weighted sum over central pixel gradients
  - that vector is transformed from a sum of gradients to a shift that would cause that difference in luminosity of the pixels. 
- corners attempts to find new tracking points

Once these pipelines are run the tracking state is updated in the following fashion:
- .


Q: What is KLT?

Q: What is a KLT photometric residual?
