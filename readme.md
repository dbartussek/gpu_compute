Notes on building:
There are three features defined in Cargo.toml

- fill_rectangle: enables the Nvidia fill rectangle shading mode. PolygonMode::FillRectangle is for some reason commented out in the vulkano source code. To enable it, the file .cargo/registry/src/index.crates.io-HASH/vulkano-VERSION/src/pipeline/graphics/rasterization has to be edited and FILL_RECTANGLE_NV uncommented
- cuda: builds the src/cuda/cuda_sum.cu file using the CUDA compiler and enables CUDA benchmarks
- opencl: enables OpenCL benchmarks. Required due to the Steam Deck not supporting OpenCL