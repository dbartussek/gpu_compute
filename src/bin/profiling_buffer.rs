#![feature(int_roundings)]

use gpu_compute::{execute_util_compute::ComputeExecuteUtil, vulkan_util::VulkanData};
use nalgebra::Vector2;

fn main() {
    let mut vulkan = VulkanData::init();

    let data_size = Vector2::<u32>::new(256 * 32, 100_000_000u32.div_ceil(32 * 256));

    let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
    let mut execute = ComputeExecuteUtil::setup_storage_buffer(
        &mut vulkan,
        data_size,
        &shader,
        compute_none_sbuffer_loop::SpecializationConstants {
            TEXTURE_SIZE_X: data_size.x as _,
            TEXTURE_SIZE_Y: 1,
        },
        1,
    );

    execute.run(&mut vulkan, true);
}

mod compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}
