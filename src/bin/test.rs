use gpu_compute::{capture::capture, execute_util::ExecuteUtil, vulkan_util::VulkanData};

const DATA_SIZE: usize = 1;

fn main() {
    let mut vulkan = VulkanData::init();

    let shader = attach_discard_sampled_many::load(vulkan.device.clone()).unwrap();
    let mut execute = ExecuteUtil::setup_1d_sampler(
        &mut vulkan,
        DATA_SIZE,
        &shader,
        attach_discard_sampled_many::SpecializationConstants {
            TEXTURE_SIZE_X: DATA_SIZE as _,
            TEXTURE_SIZE_Y: 1,
        },
    );

    capture(|| {
        execute.run(&mut vulkan);
    });
}

mod attach_discard_sampled_many {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/attach_discard_sampled_many.glsl",
        include: ["shaders/pluggable"],
    }
}

mod attach_discard_sbuffer_many {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/attach_discard_sbuffer_many.glsl",
        include: ["shaders/pluggable"],
    }
}
