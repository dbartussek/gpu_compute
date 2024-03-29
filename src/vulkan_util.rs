use bytemuck::{Pod, Zeroable};
use itertools::Itertools;
use nalgebra::Vector2;
use smallvec::smallvec;
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BufferImageCopy,
        CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, CopyImageToBufferInfo,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Features, Queue, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{
        AttachmentImage, ImageAccess, ImageCreateFlags, ImageDimensions, ImageSubresourceLayers,
        ImageUsage, ImmutableImage, MipmapsCount, StorageImage,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
    render_pass::RenderPass,
    single_pass_renderpass, Version, VulkanLibrary,
};

pub struct VulkanData {
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub queue_compute: Arc<Queue>,
    pub memory_allocator: StandardMemoryAllocator,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub descriptor_set_allocator: StandardDescriptorSetAllocator,

    render_pass_cache: HashMap<RenderPassKey, Arc<RenderPass>>,
    vertex_buffer: Subbuffer<[MVertex]>,

    pub supports_fill_rectangle: bool,

    max_size: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderPassKey {
    pub format: Option<Format>,
}

#[derive(Pod, Zeroable, Copy, Clone, Vertex)]
#[repr(C)]
pub struct MVertex {
    #[format(R8_UINT)]
    dummy: u8,
}

impl VulkanData {
    pub fn init() -> Self {
        let library = VulkanLibrary::new().unwrap();

        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                max_api_version: Some(Version::V1_2),
                enabled_extensions: vulkano_win::required_extensions(&library),
                ..Default::default()
            },
        )
        .unwrap();

        let (physical_device, queue_family_index, queue_compute) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::GRAPHICS))
                    .map(|i| (p, i as u32))
                    .and_then(|(p, graphics)| {
                        p.queue_family_properties()
                            .iter()
                            .position(|q| {
                                !q.queue_flags.intersects(QueueFlags::GRAPHICS)
                                    && q.queue_flags.intersects(QueueFlags::COMPUTE)
                            })
                            .or_else(|| {
                                p.queue_family_properties()
                                    .iter()
                                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                            })
                            .map(|compute| (p, graphics, compute as u32))
                    })
            })
            .min_by_key(|(p, _, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        dbg!(physical_device.properties().device_type);
        dbg!(physical_device.supported_extensions().nv_fill_rectangle);

        let queues = if queue_family_index != queue_compute {
            vec![
                QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: queue_compute,
                    ..Default::default()
                },
            ]
        } else {
            vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }]
        };

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    nv_fill_rectangle: physical_device.supported_extensions().nv_fill_rectangle,
                    khr_swapchain: true,
                    ..Default::default()
                },
                enabled_features: Features {
                    fill_mode_non_solid: true,
                    ..Default::default()
                },
                queue_create_infos: queues,
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();
        let queue_compute = if queue_family_index != queue_compute {
            queues.next().unwrap()
        } else {
            queue.clone()
        };

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let vertices = [MVertex { dummy: 0 }; 4];
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

        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

        let max_size = physical_device
            .properties()
            .max_image_dimension2_d
            .min(physical_device.properties().max_viewport_dimensions[0]);
        println!(
            "Maximum render size: {:?}",
            physical_device.properties().max_viewport_dimensions
        );
        println!(
            "Maximum 2d image size: {}",
            physical_device.properties().max_image_dimension2_d
        );
        println!("Maximum overall size: {}", max_size);

        Self {
            instance,
            physical_device: physical_device.clone(),
            device,
            queue,
            queue_compute,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            render_pass_cache: Default::default(),
            vertex_buffer,
            supports_fill_rectangle: physical_device.supported_extensions().nv_fill_rectangle,
            max_size,
        }
    }

    pub fn profiling_sizes(&self) -> Vec<u32> {
        [1u32]
            .into_iter()
            // .chain((0..=(50_000 * 4)).step_by(256 * 32 * 32))
            // .chain(((50_000 * 4)..=(50_000 * 64)).step_by(256 * 256 * 10))
            // .chain(((0)..=(50_000 * 64 * 64)).step_by(256 * 256 * 12 * 64))
            .chain((0..29).step_by(2).map(|l| 1 << l))
            .chain([1 << 29])
            .filter(|v| *v != 0)
            .map(|v| v.div_ceil(self.max_size) * self.max_size)
            .unique()
            .sorted()
            .collect_vec()
    }

    pub fn gpu_thread_count(&self) -> u32 {
        self.max_size
    }

    pub fn create_command_buffer(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    pub fn create_target_image(&self, size: Vector2<u32>, format: Format) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            &self.memory_allocator,
            size.into(),
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
                    if let Some(format) = key.format {
                        single_pass_renderpass!(self.device.clone(),
                            attachments: {
                                color: {
                                    load: Clear,
                                    store: Store,
                                    format: format,
                                    samples: 1,
                                },
                            },
                            pass: {
                                color: [color],
                                depth_stencil: {},
                            },
                        )
                    } else {
                        single_pass_renderpass!(self.device.clone(),
                            attachments: {},
                            pass: {
                                color: [],
                                depth_stencil: {},
                            },
                        )
                    }
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

    pub fn create_1d_data_sample_image<Px, I>(
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

    pub fn create_2d_data_sample_image<Px, I>(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        size: Vector2<u32>,
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

        ImmutableImage::from_iter(
            &self.memory_allocator,
            iter,
            ImageDimensions::Dim2d {
                width: size.x,
                height: size.y,
                array_layers: 1,
            },
            MipmapsCount::One,
            format,
            command_buffer,
        )
        .unwrap()
    }

    pub fn create_storage_buffer<T, I>(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        iter: I,
    ) -> Subbuffer<[T]>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let staging = Buffer::from_iter(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            iter,
        )
        .unwrap();

        let buffer = Buffer::new_slice(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            staging.len(),
        )
        .unwrap();

        command_buffer
            .copy_buffer(CopyBufferInfo::buffers(staging, buffer.clone()))
            .unwrap();

        buffer
    }

    pub fn create_1d_data_storage_image<Px, I>(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        iter: I,
        format: Format,
    ) -> Arc<StorageImage>
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

        let source = Buffer::from_iter(
            &self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            iter,
        )
        .unwrap();

        let image = StorageImage::with_usage(
            &self.memory_allocator,
            ImageDimensions::Dim1d {
                width: size as _,
                array_layers: 1,
            },
            format,
            ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ImageCreateFlags::empty(),
            source
                .device()
                .active_queue_family_indices()
                .iter()
                .copied(),
        )
        .unwrap();

        let region = BufferImageCopy {
            image_subresource: ImageSubresourceLayers::from_parameters(
                format,
                image.dimensions().array_layers(),
            ),
            image_extent: image.dimensions().width_height_depth(),
            ..Default::default()
        };
        command_buffer
            .copy_buffer_to_image(CopyBufferToImageInfo {
                regions: smallvec![region],
                ..CopyBufferToImageInfo::buffer_image(source, image.clone())
            })
            .unwrap();

        image
    }
}
