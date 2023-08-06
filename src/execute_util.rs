use crate::vulkan_util::{MVertex, RenderPassKey, VulkanData};
use itertools::Itertools;
use nalgebra::Vector2;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::{hint::black_box, iter::once, num::Wrapping, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        CopyBufferInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    format::{ClearValue, Format},
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            rasterization::{PolygonMode, RasterizationState},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Sampler, SamplerCreateInfo},
    shader::{ShaderModule, SpecializationConstants},
    sync::GpuFuture,
    DeviceSize,
};

pub struct ExecuteUtil {
    viewport_size: Vector2<u32>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<PersistentDescriptorSet>,

    instance_id: u32,

    expected_result: u32,

    output: OutputKind,
}

pub fn generate_data(length: u32) -> impl ExactSizeIterator<Item = u32> {
    let mut rng = Pcg64Mcg::seed_from_u64(42);

    (1u32..(length + 1)).map(move |_| rng.gen_range(0..=1))
}

pub fn s32<It>(it: It) -> u32
where
    It: IntoIterator<Item = u32>,
{
    it.into_iter().map(|s| Wrapping(s)).sum::<Wrapping<u32>>().0
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub enum OutputKind {
    Attachment,
    Buffer,
}

impl OutputKind {
    fn to_render_pass_key(&self) -> RenderPassKey {
        RenderPassKey {
            format: match self {
                OutputKind::Attachment => Some(Format::R32_UINT),
                OutputKind::Buffer => None,
            },
        }
    }
}

impl ExecuteUtil {
    #[inline(always)]
    fn generic_setup<SC, INIT>(
        vulkan: &mut VulkanData,
        fs: &ShaderModule,
        sc: SC,
        output: OutputKind,

        specialized_init: INIT,
    ) -> Self
    where
        SC: SpecializationConstants,
        INIT: FnOnce(
            &mut VulkanData,
            &Arc<GraphicsPipeline>,
        ) -> (Vector2<u32>, Arc<PersistentDescriptorSet>, u32),
    {
        let render_pass = vulkan.create_render_pass(output.to_render_pass_key());

        let vert = vs::load(vulkan.device.clone()).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(MVertex::per_vertex())
            .vertex_shader(vert.entry_point("main").unwrap(), vs::SpecializationConstants{DATA_SCALE: 1})
            // .rasterization_state(RasterizationState::new().polygon_mode(PolygonMode::FillRectangle))
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), sc)
            .color_blend_state(
                // TODO make this generic?
                ColorBlendState::new(subpass.num_color_attachments())
                    // .blend(AttachmentBlend::additive()),
            )
            .render_pass(subpass)
            .build(vulkan.device.clone())
            .unwrap();

        let (viewport_size, set, expected_result) = specialized_init(vulkan, &pipeline);

        Self {
            viewport_size,
            render_pass,
            pipeline,
            set,
            instance_id: 1,
            expected_result,
            output,
        }
    }

    #[inline(always)]
    pub fn setup_storage_buffer<SC>(
        vulkan: &mut VulkanData,
        data_size: Vector2<u32>,
        fs: &ShaderModule,
        sc: SC,
        output: OutputKind,

        framebuffer_y: u32,
        vectorization_factor: u32,
    ) -> Self
    where
        SC: SpecializationConstants,
    {
        assert_eq!(data_size.x % framebuffer_y, 0);

        let mut executor = Self::generic_setup(vulkan, fs, sc, output, |vulkan, pipeline| {
            let total = data_size.x * data_size.y;

            let generated_data = generate_data(total).collect_vec();

            let mut command_buffer = vulkan.create_command_buffer();
            let data =
                vulkan.create_storage_buffer(&mut command_buffer, generated_data.iter().copied());

            command_buffer
                .build()
                .unwrap()
                .execute(vulkan.queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            let set = PersistentDescriptorSet::new(
                &vulkan.descriptor_set_allocator,
                pipeline.layout().set_layouts().get(0).unwrap().clone(),
                [WriteDescriptorSet::buffer(0, data)],
            )
            .unwrap();

            (
                Vector2::new(data_size.x / framebuffer_y, framebuffer_y),
                set,
                s32(generated_data),
            )
        });

        assert_eq!(
            data_size.y % vectorization_factor,
            0,
            "Dimension Y must be a multiple of the vectorization factor"
        );
        executor.instance_id = data_size.y / vectorization_factor;

        executor
    }

    #[inline(always)]
    pub fn setup_1d_sampler<SC>(
        vulkan: &mut VulkanData,
        data_size: usize,
        fs: &ShaderModule,
        sc: SC,
        output: OutputKind,
    ) -> Self
    where
        SC: SpecializationConstants,
    {
        Self::generic_setup(vulkan, fs, sc, output, |vulkan, pipeline| {
            let mut command_buffer = vulkan.create_command_buffer();

            let raw_data = generate_data(data_size as u32).collect_vec();

            let data = vulkan.create_1d_data_sample_image(
                &mut command_buffer,
                raw_data.iter().copied(),
                Format::R32_UINT,
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

            (Vector2::new(data_size as _, 1), set, s32(raw_data))
        })
    }

    #[inline(always)]
    fn run_for_attachment(&mut self, vulkan: &mut VulkanData) {
        let mut command_buffer = vulkan.create_command_buffer();

        let target = vulkan.create_target_image(self.viewport_size, Format::R32_UINT);
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
            .draw(vulkan.vertex_buffer().len() as _, 1, 0, self.instance_id)
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

        // dbg!(&read_buffer.read().unwrap() as &[_]);

        let result = black_box(s32(read_buffer.read().unwrap().iter().copied()));
        assert_eq!(result, self.expected_result)
    }

    #[inline(always)]
    fn run_for_buffer(&mut self, vulkan: &mut VulkanData, separate_read_buffer: bool) {
        let mut command_buffer = vulkan.create_command_buffer();

        let target: Subbuffer<[u32]> = Buffer::new_slice(
            &vulkan.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: if separate_read_buffer {
                    MemoryUsage::DeviceOnly
                } else {
                    MemoryUsage::Download
                },
                ..Default::default()
            },
            (self.viewport_size.x * self.viewport_size.y) as DeviceSize,
        )
        .unwrap();
        let read_buffer = if separate_read_buffer {
            Buffer::new_slice(
                &vulkan.memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Download,
                    ..Default::default()
                },
                target.len(),
            )
            .unwrap()
        } else {
            target.clone()
        };

        let target_set = PersistentDescriptorSet::new(
            &vulkan.descriptor_set_allocator,
            self.pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, target.clone())],
        )
        .unwrap();

        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                extent: self.viewport_size.into(),
                layers: 1,
                ..Default::default()
            },
        )
        .unwrap();

        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo::framebuffer(framebuffer),
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
            ).bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                1,
                target_set,
            )
            .bind_vertex_buffers(0, vulkan.vertex_buffer())
            .draw(vulkan.vertex_buffer().len() as _, 1, 0, self.instance_id)
            .unwrap()

            // End rendering
            .end_render_pass()
            .unwrap();

        if separate_read_buffer {
            command_buffer
                .copy_buffer(CopyBufferInfo::buffers(target, read_buffer.clone()))
                .unwrap();
        }

        let future = command_buffer
            .build()
            .unwrap()
            .execute(vulkan.queue.clone())
            .unwrap();
        let fence = future.then_signal_fence_and_flush().unwrap();
        fence.wait(None).unwrap();

        // dbg!(&read_buffer.read().unwrap() as &[_]);

        let result = black_box(s32(read_buffer.read().unwrap().iter().copied()));
        assert_eq!(result, self.expected_result)
    }

    #[inline(always)]
    pub fn run(&mut self, vulkan: &mut VulkanData, separate_read_buffer: bool) {
        match self.output {
            OutputKind::Attachment => self.run_for_attachment(vulkan),
            OutputKind::Buffer => self.run_for_buffer(vulkan, separate_read_buffer),
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/basic.vs",
    }
}
