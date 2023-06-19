use gpu_compute::vulkan_util::{MVertex, RenderPassKey, VulkanData};
use itertools::Itertools;
use lazy_static::lazy_static;
use nalgebra::Vector2;
use renderdoc::{RenderDoc, V110};
use std::{ffi::c_void, iter::once, ops::Deref, ptr::null, sync::Mutex};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    format::{ClearValue, Format},
    image::{view::ImageView, ImageAccess},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::GpuFuture,
};

const DATA_SIZE: u32 = 1024;

fn main() {
    let mut vulkan = VulkanData::init();

    capture(|| {
        let mut command_buffer = vulkan.create_command_buffer();

        let target = vulkan.create_target_image(Vector2::new(DATA_SIZE, 1), Format::R32_UINT);
        let render_pass = vulkan.create_render_pass(RenderPassKey {
            format: Some(target.format()),
        });

        let framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![ImageView::new_default(target.clone()).unwrap()],
                ..Default::default()
            },
        )
        .unwrap();

        let vert = vs::load(vulkan.device.clone()).unwrap();
        let frag = fs::load(vulkan.device.clone()).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(MVertex::per_vertex())
            .vertex_shader(vert.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(frag.entry_point("main").unwrap(), ())
            .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()))
            .render_pass(subpass)
            .build(vulkan.device.clone())
            .unwrap();

        let data = vulkan.create_1d_data_storage_image(
            &mut command_buffer,
            (0u32..(DATA_SIZE as _)).into_iter().collect_vec(),
            Format::R32_UINT,
        );

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(vulkan.device.clone());
        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::image_view(
                0,
                ImageView::new_default(data).unwrap(),
            )],
        )
        .unwrap();

        command_buffer
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(ClearValue::Uint([0; 4]))],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassContents::Inline,
        )
        .unwrap()

        // Actual rendering
        .set_viewport(0, once(Viewport{
            origin: [0.0, 0.0],
            dimensions: [DATA_SIZE as f32, 1.0],
            depth_range: 0.0..1.0
        }))
        .bind_pipeline_graphics(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            set,
        )
        .bind_vertex_buffers(0, vulkan.vertex_buffer())
        .draw(vulkan.vertex_buffer().len() as _, 1, 0, 0)
        .unwrap()

        // End rendering
        .end_render_pass()
        .unwrap();

        let read_buffer: Subbuffer<[u32]> =
            vulkan.download_image(&mut command_buffer, target.clone());

        let future = command_buffer
            .build()
            .unwrap()
            .execute(vulkan.queue.clone())
            .unwrap();
        let fence = future.then_signal_fence_and_flush().unwrap();
        fence.wait(None).unwrap();

        let data = read_buffer.read().unwrap();
        let data = data.deref();
        // println!("{:?}", &data[..(DATA_SIZE as usize / 2)]);
        // println!("{:?}", &data[(DATA_SIZE as usize / 2)..]);

        for (index, (a, b)) in data.iter().copied().zip(0..).enumerate() {
            assert_eq!(a, b, "mismatch at {}", index);
        }
    });
}

fn capture<F, R>(function: F) -> R
where
    F: FnOnce() -> R,
{
    lazy_static! {
        static ref RENDERDOC: Option<Mutex<RenderDoc<V110>>> =
            RenderDoc::<V110>::new().ok().map(Mutex::new);
    }

    let mut renderdoc = (*RENDERDOC).as_ref().map(|r| r.lock().unwrap());

    if let Some(doc) = renderdoc.as_mut() {
        doc.start_frame_capture(null::<c_void>(), null::<c_void>());
    }

    let result = function();

    if let Some(doc) = renderdoc.as_mut() {
        doc.end_frame_capture(null::<c_void>(), null::<c_void>());
    }

    result
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/basic.vs",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/basic_image_access.fs",
    }
}
