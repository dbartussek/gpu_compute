use crate::vulkan_util::{MVertex, RenderPassKey, VulkanData};
use itertools::Itertools;
use nalgebra::Vector2;
use std::{iter::once, sync::Arc};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    format::{ClearValue, Format::R32_UINT},
    image::view::ImageView,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Sampler, SamplerCreateInfo},
    shader::{ShaderModule, SpecializationConstants},
    sync::GpuFuture,
};

pub struct ExecuteUtil {
    viewport_size: Vector2<u32>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<PersistentDescriptorSet>,
}

impl ExecuteUtil {
    #[inline(always)]
    fn generic_setup<SC, INIT>(
        vulkan: &mut VulkanData,
        fs: &ShaderModule,
        sc: SC,
        key: RenderPassKey,

        specialized_init: INIT,
    ) -> Self
    where
        SC: SpecializationConstants,
        INIT: FnOnce(
            &mut VulkanData,
            &Arc<GraphicsPipeline>,
        ) -> (Vector2<u32>, Arc<PersistentDescriptorSet>),
    {
        let render_pass = vulkan.create_render_pass(key);

        let vert = vs::load(vulkan.device.clone()).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(MVertex::per_vertex())
            .vertex_shader(vert.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), sc)
            .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()))
            .render_pass(subpass)
            .build(vulkan.device.clone())
            .unwrap();

        let (viewport_size, set) = specialized_init(vulkan, &pipeline);

        Self {
            viewport_size,
            render_pass,
            pipeline,
            set,
        }
    }

    #[inline(always)]
    pub fn setup_1d_sampler<SC>(vulkan: &mut VulkanData, data_size: usize, fs: &ShaderModule, sc: SC) -> Self
    where
        SC: SpecializationConstants,
    {
        Self::generic_setup(
            vulkan,
            fs,
            sc,
            RenderPassKey {
                format: Some(R32_UINT),
            },
            |vulkan, pipeline| {
                let mut command_buffer = vulkan.create_command_buffer();

                let data = vulkan.create_1d_data_sample_image(
                    &mut command_buffer,
                    (1u32..=(data_size as _)).into_iter().collect_vec(),
                    R32_UINT,
                );

                command_buffer
                    .build()
                    .unwrap()
                    .execute(vulkan.queue.clone())
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();

                let descriptor_set_allocator =
                    StandardDescriptorSetAllocator::new(vulkan.device.clone());
                let sampler = Sampler::new(
                    vulkan.device.clone(),
                    SamplerCreateInfo {
                        unnormalized_coordinates: true,
                        ..Default::default()
                    },
                )
                .unwrap();

                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    pipeline.layout().set_layouts().get(0).unwrap().clone(),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new_default(data).unwrap(),
                        sampler,
                    )],
                )
                .unwrap();

                (Vector2::new(data_size as _, 1), set)
            },
        )
    }

    #[inline(always)]
    pub fn run(&mut self, vulkan: &mut VulkanData) {
        let mut command_buffer = vulkan.create_command_buffer();

        let target = vulkan.create_target_image(self.viewport_size, R32_UINT);
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![ImageView::new_default(target.clone()).unwrap()],
                ..Default::default()
            },
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
            .set_viewport(0, once(Viewport {
                origin: [0.0, 0.0],
                dimensions: self.viewport_size.map(|e| e as f32).into(),
                depth_range: 0.0..1.0
            }))
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.set.clone(),
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

        read_buffer.read().unwrap();
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/basic.vs",
    }
}
