use crate::vulkan_util::{MVertex, RenderPassKey, VulkanData};
use bytemuck::Pod;
use derivative::Derivative;
use itertools::Itertools;
use nalgebra::Vector2;
use num::{NumCast, Zero};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::{
    fmt::Debug,
    hint::black_box,
    iter::{once, Sum},
    sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
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
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
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

pub fn generate_data<Type>(length: u32) -> impl ExactSizeIterator<Item = Type>
where
    Type: NumCast,
{
    let mut rng = Pcg64Mcg::seed_from_u64(42);

    (1u32..(length + 1))
        .map(move |_| rng.gen_range(0..10))
        .map(|n| Type::from(n).unwrap())
}


pub struct ExecuteUtil<Type> {
    viewport_size: Vector2<u32>,
    parameters: ExecuteParameters,

    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<PersistentDescriptorSet>,

    instance_id: u32,

    expected_result: Type,

    accumulate: Box<dyn Fn(Type, Type) -> Type>,

    data_size: u32,
}


#[derive(Derivative)]
#[derivative(Default, Clone)]
pub struct ExecuteParameters {
    #[derivative(Default(value = "1"))]
    pub vectorization_factor: u32,

    #[derivative(Default(value = "1"))]
    pub framebuffer_y: u32,

    #[derivative(Default(value = "ClearValue::Uint([0; 4])"))]
    pub clear_value: ClearValue,

    pub output: OutputKind,
    pub quad_method: QuadMethod,

    pub blend: Option<BlendMethod>,
    pub use_instances_and_blend: bool,
}


#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
#[allow(non_camel_case_types)]
pub enum BlendMethod {
    Add,
    Min,
}

impl BlendMethod {
    pub fn to_vulkano(self) -> AttachmentBlend {
        let mut base = AttachmentBlend {
            color_op: BlendOp::ReverseSubtract,
            color_source: BlendFactor::One,
            color_destination: BlendFactor::One,
            alpha_op: BlendOp::ReverseSubtract,
            alpha_source: BlendFactor::One,
            alpha_destination: BlendFactor::One,
        };

        match self {
            BlendMethod::Add => {
                base.color_op = BlendOp::Add;
                base.alpha_op = BlendOp::Add;
            },
            BlendMethod::Min => {
                base.color_op = BlendOp::Min;
                base.alpha_op = BlendOp::Min;
            },
        }

        base
    }
}

#[derive(Derivative)]
#[derivative(Default)]
#[derive(Copy, Clone, Debug)]
pub enum OutputKind {
    RenderAttachment(Format),
    #[derivative(Default)]
    Buffer,
}

impl OutputKind {
    #[allow(non_upper_case_globals)]
    pub const Attachment: Self = Self::RenderAttachment(Format::R32_UINT);

    fn to_render_pass_key(self) -> RenderPassKey {
        RenderPassKey {
            format: match self {
                OutputKind::RenderAttachment(format) => Some(format),
                OutputKind::Buffer => None,
            },
        }
    }
}

#[derive(Derivative)]
#[derivative(Default)]
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug, Hash)]
#[allow(non_camel_case_types)]
pub enum QuadMethod {
    two_triangles,
    #[derivative(Default)]
    large_triangle,
    #[cfg(feature = "fill_rectangle")]
    fill_rectangle,
}

impl QuadMethod {
    pub fn all(vulkan: &VulkanData) -> &'static [QuadMethod] {
        static BASIC: &[QuadMethod] = &[QuadMethod::two_triangles, QuadMethod::large_triangle];
        static WITH_RECTANGLE: &[QuadMethod] = &[
            QuadMethod::two_triangles,
            QuadMethod::large_triangle,
            #[cfg(feature = "fill_rectangle")]
            QuadMethod::fill_rectangle,
        ];

        if vulkan.supports_fill_rectangle {
            WITH_RECTANGLE
        } else {
            BASIC
        }
    }
}

impl<Type> ExecuteUtil<Type>
where
    Type: Copy + NumCast + Pod + BufferContents + PartialEq + Zero + Debug + Sum,
{
    #[inline(always)]
    fn generic_setup<SC, INIT, Acc>(
        vulkan: &mut VulkanData,
        fs: &ShaderModule,
        sc: SC,
        params: ExecuteParameters,
        data_size: u32,

        accumulate: Acc,

        specialized_init: INIT,
    ) -> Self
    where
        SC: SpecializationConstants,
        Acc: 'static + Fn(Type, Type) -> Type,
        INIT: FnOnce(
            &mut VulkanData,
            &Arc<GraphicsPipeline>,
        ) -> (Vector2<u32>, Arc<PersistentDescriptorSet>, Type),
    {
        let render_pass = vulkan.create_render_pass(params.output.to_render_pass_key());

        let vert = vs::load(vulkan.device.clone()).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let mut pipeline = GraphicsPipeline::start()
            .vertex_input_state(MVertex::per_vertex())
            .vertex_shader(
                vert.entry_point("main").unwrap(),
                vs::SpecializationConstants {
                    DATA_SCALE: if params.quad_method == QuadMethod::large_triangle {
                        2
                    } else {
                        1
                    },
                },
            )
            .rasterization_state(RasterizationState::new().polygon_mode({
                #[cfg(feature = "fill_rectangle")]
                if params.quad_method == QuadMethod::fill_rectangle {
                    PolygonMode::FillRectangle
                } else {
                    PolygonMode::Fill
                }
                #[cfg(not(feature = "fill_rectangle"))]
                PolygonMode::Fill
            }))
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), sc);
        if let Some(blend) = params.blend {
            pipeline = pipeline.color_blend_state(
                ColorBlendState::new(subpass.num_color_attachments()).blend(blend.to_vulkano()),
            );
        }
        let pipeline = pipeline
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
            parameters: params,
            accumulate: Box::new(accumulate),
            data_size,
        }
    }

    #[inline(always)]
    pub fn setup_storage_buffer<SC, Acc>(
        vulkan: &mut VulkanData,
        data_size: Vector2<u32>,
        fs: &ShaderModule,
        sc: SC,
        params: ExecuteParameters,

        accumulate: Acc,
    ) -> Self
    where
        SC: SpecializationConstants,
        Acc: 'static + Fn(Type, Type) -> Type,
    {
        assert_eq!(data_size.x % params.framebuffer_y, 0);

        let total = data_size.x * data_size.y;
        let generated_data = generate_data(total).collect_vec();
        let expected = generated_data.iter().copied().reduce(&accumulate).unwrap();

        let mut executor = Self::generic_setup(
            vulkan,
            fs,
            sc,
            params.clone(),
            total,
            accumulate,
            move |vulkan, pipeline| {
                let mut command_buffer = vulkan.create_command_buffer();
                let data = vulkan
                    .create_storage_buffer(&mut command_buffer, generated_data.iter().copied());

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
                    Vector2::new(data_size.x / params.framebuffer_y, params.framebuffer_y),
                    set,
                    expected,
                )
            },
        );

        assert_eq!(
            data_size.y % params.vectorization_factor,
            0,
            "Dimension Y must be a multiple of the vectorization factor"
        );
        executor.instance_id = data_size.y / params.vectorization_factor;

        executor
    }

    #[inline(always)]
    pub fn setup_2d_sampler<SC, Acc>(
        vulkan: &mut VulkanData,
        data_size: Vector2<u32>,
        fs: &ShaderModule,
        sc: SC,
        params: ExecuteParameters,

        accumulate: Acc,
    ) -> Self
    where
        SC: SpecializationConstants,
        Acc: 'static + Fn(Type, Type) -> Type,
    {
        let total = data_size.x * data_size.y;
        let raw_data = generate_data(total).collect_vec();
        let expected = raw_data.iter().copied().reduce(&accumulate).unwrap();

        let mut executor = Self::generic_setup(
            vulkan,
            fs,
            sc,
            params.clone(),
            total,
            accumulate,
            |vulkan, pipeline| {
                let mut command_buffer = vulkan.create_command_buffer();


                let data = vulkan.create_2d_data_sample_image(
                    &mut command_buffer,
                    data_size,
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

                (
                    Vector2::new(data_size.x / params.framebuffer_y, params.framebuffer_y),
                    set,
                    expected,
                )
            },
        );

        executor.instance_id = data_size.y;

        executor
    }

    #[inline(always)]
    fn run_for_attachment(&mut self, vulkan: &mut VulkanData, format: Format) {
        let mut command_buffer = vulkan.create_command_buffer();

        let target = vulkan.create_target_image(self.viewport_size, format);
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
                    clear_values: vec![Some(self.parameters.clear_value)],
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
            .push_constants(self.pipeline.layout().clone(), 0, self.data_size)
            .draw(
                if self.parameters.quad_method == QuadMethod::two_triangles {vulkan.vertex_buffer().len() as _} else {3},
                if self.parameters.use_instances_and_blend {self.instance_id} else {1},
                0,
                if self.parameters.use_instances_and_blend {0} else {self.instance_id}
            )
            .unwrap()

            // End rendering
            .end_render_pass()
            .unwrap();

        let read_buffer: Subbuffer<[Type]> =
            vulkan.download_image(&mut command_buffer, target.clone());

        let future = command_buffer
            .build()
            .unwrap()
            .execute(vulkan.queue.clone())
            .unwrap();
        let fence = future.then_signal_fence_and_flush().unwrap();
        fence.wait(None).unwrap();

        // dbg!(&read_buffer.read().unwrap() as &[_]);

        let result = black_box(
            read_buffer
                .read()
                .unwrap()
                .iter()
                .copied()
                .reduce(&self.accumulate)
                .unwrap(),
        );
        assert_eq!(result, self.expected_result);
        // dbg!(result, self.expected_result);
    }

    #[inline(always)]
    fn run_for_buffer(&mut self, vulkan: &mut VulkanData, separate_read_buffer: bool) {
        let mut command_buffer = vulkan.create_command_buffer();

        let target: Subbuffer<[Type]> = Buffer::new_slice(
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
            (self.viewport_size.x * self.viewport_size.y * self.parameters.vectorization_factor)
                as DeviceSize,
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
            .push_constants(self.pipeline.layout().clone(), 0, self.data_size)
            .draw(if self.parameters.quad_method == QuadMethod::two_triangles {vulkan.vertex_buffer().len() as _} else {3}, 1, 0, self.instance_id)
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

        let result = black_box(
            read_buffer
                .read()
                .unwrap()
                .iter()
                .copied()
                .reduce(&self.accumulate)
                .unwrap(),
        );
        // assert_eq!(result, self.expected_result);
    }

    #[inline(always)]
    pub fn run(&mut self, vulkan: &mut VulkanData, separate_read_buffer: bool) {
        match self.parameters.output {
            OutputKind::RenderAttachment(format) => self.run_for_attachment(vulkan, format),
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
