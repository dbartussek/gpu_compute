use gpu_compute::{
    capture::capture,
    execute_util::{ExecuteUtil, OutputKind},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;

fn main() {
    let mut vulkan = VulkanData::init();

    let shader = attach_discard_sbuffer_many::load(vulkan.device.clone()).unwrap();
    let mut execute = ExecuteUtil::setup_storage_buffer(
        &mut vulkan,
        Vector2::new(256, 2),
        &shader,
        attach_discard_sbuffer_many::SpecializationConstants {
            TEXTURE_SIZE_X: 256,
            TEXTURE_SIZE_Y: 1,
        },
        OutputKind::Attachment,
    );

    capture(|| {
        execute.run(&mut vulkan);
    });
}

mod attach_discard_sbuffer_many {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/attach_discard_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}
