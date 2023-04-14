use itertools::Itertools;
use lazy_static::lazy_static;
use renderdoc::{RenderDoc, V110};
use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::c_void,
    iter::once,
    ops::Deref,
    ptr::null,
    sync::{Arc, Mutex},
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageToBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    format::{ClearValue, Format},
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage,
        MipmapsCount,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
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
    single_pass_renderpass,
    sync::GpuFuture,
    VulkanLibrary,
};

const DATA_SIZE: u32 = 1024;

pub struct VulkanData {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: StandardMemoryAllocator,
    pub command_buffer_allocator: StandardCommandBufferAllocator,

    render_pass_cache: HashMap<RenderPassKey, Arc<RenderPass>>,
    vertex_buffer: Subbuffer<[MVertex]>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderPassKey {
    pub format: Format,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct MVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl VulkanData {
    fn init() -> Self {
        let library = VulkanLibrary::new().unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enumerate_portability: true,
                ..Default::default()
            },
        )
        .unwrap();

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: Default::default(),
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let vertices = [
            MVertex {
                position: [-1.0, -1.0],
            },
            MVertex {
                position: [-1.0, 1.0],
            },
            MVertex {
                position: [1.0, -1.0],
            },
            MVertex {
                position: [1.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            render_pass_cache: Default::default(),
            vertex_buffer,
        }
    }

    pub fn create_command_buffer(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    pub fn create_target_image(&self, size: u32, format: Format) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            &self.memory_allocator,
            [size, 1],
            format,
            ImageUsage::TRANSFER_SRC,
        )
        .unwrap()
    }

    pub fn create_render_pass(&mut self, key: RenderPassKey) -> Arc<RenderPass> {
        match self.render_pass_cache.entry(key) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => e
                .insert(
                    single_pass_renderpass!(self.device.clone(),
                        attachments: {
                            color: {
                                load: Clear,
                                store: Store,
                                format: key.format,
                                samples: 1,
                            },
                        },
                        pass: {
                            color: [color],
                            depth_stencil: {},
                        },
                    )
                    .unwrap(),
                )
                .clone(),
        }
    }

    pub fn vertex_buffer(&self) -> Subbuffer<[MVertex]> {
        self.vertex_buffer.clone()
    }

    pub fn download_image<Px, I>(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image: Arc<I>,
    ) -> Subbuffer<[Px]>
    where
        Px: BufferContents,
        I: ImageAccess + 'static,
    {
        let pixel_bits = image
            .format()
            .components()
            .iter()
            .copied()
            .map(|i| i as usize)
            .sum::<usize>();
        assert_eq!(pixel_bits, std::mem::size_of::<Px>() * 8);

        let read_buffer = Buffer::new_slice::<Px>(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Download,
                ..Default::default()
            },
            match image.dimensions() {
                ImageDimensions::Dim1d {
                    width,
                    array_layers,
                } => width * array_layers,
                ImageDimensions::Dim2d {
                    width,
                    height,
                    array_layers,
                } => width * height * array_layers,
                ImageDimensions::Dim3d {
                    width,
                    height,
                    depth,
                } => width * height * depth,
            } as u64,
        )
        .unwrap();

        command_buffer
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image,
                read_buffer.clone(),
            ))
            .unwrap();

        read_buffer
    }

    pub fn create_1d_data_image<Px, I>(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        iter: I,
        format: Format,
    ) -> Arc<ImmutableImage>
    where
        Px: BufferContents,
        I: IntoIterator<Item = Px>,
        I::IntoIter: ExactSizeIterator,
    {
        let pixel_bits = format
            .components()
            .iter()
            .copied()
            .map(|i| i as usize)
            .sum::<usize>();
        assert_eq!(pixel_bits, std::mem::size_of::<Px>() * 8);

        let iter = iter.into_iter();
        let size = iter.len();

        ImmutableImage::from_iter(
            &self.memory_allocator,
            iter,
            ImageDimensions::Dim1d {
                width: size as _,
                array_layers: 1,
            },
            MipmapsCount::One,
            format,
            command_buffer,
        )
        .unwrap()
    }
}

fn main() {
    let mut vulkan = VulkanData::init();

    capture(|| {
        let mut command_buffer = vulkan.create_command_buffer();

        let target = vulkan.create_target_image(DATA_SIZE, Format::R32_UINT);
        let render_pass = vulkan.create_render_pass(RenderPassKey {
            format: target.format(),
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

        let data = vulkan.create_1d_data_image(
            &mut command_buffer,
            (0u32..(DATA_SIZE as _)).into_iter().collect_vec(),
            Format::R32_UINT,
        );

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(vulkan.device.clone());
        let sampler = Sampler::new(vulkan.device.clone(), SamplerCreateInfo::default()).unwrap();
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

        for (a, b) in data.iter().copied().zip(0..) {
            assert_eq!(a, b);
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
        src: r"
            #version 450
            layout(location = 0) in vec2 position;
            layout(location = 0) out float tex_coord;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                tex_coord = (position.x + 1) / 2;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            layout(location = 0) in float tex_coord;
            layout(location = 0) out uvec4 f_color;
            layout(set = 0, binding = 0) uniform usampler1D tex;

            void main() {
                f_color = texture(tex, tex_coord);
            }
        ",
    }
}
