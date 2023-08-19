use crate::{execute_util::generate_data, vulkan_util::VulkanData};
use bytemuck::Pod;
use derivative::Derivative;
use itertools::Itertools;
use nalgebra::Vector2;
use num::{NumCast, Zero};
use std::{fmt::Debug, hint::black_box, iter::Sum, marker::PhantomData, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{CopyBufferInfo, PrimaryCommandBufferAbstract},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{ShaderModule, SpecializationConstants},
    sync::GpuFuture,
    DeviceSize,
};

pub struct ComputeExecuteUtil<Type> {
    viewport_size: Vector2<u32>,
    parameters: ComputeParameters,

    pipeline: Arc<ComputePipeline>,
    set: Arc<PersistentDescriptorSet>,

    instance_id: u32,

    expected_result: Type,

    accumulate: Box<dyn Fn(Type, Type) -> Type>,

    t: PhantomData<Type>,
}

#[derive(Derivative)]
#[derivative(Default, Clone)]
#[derive(Copy)]
pub enum OutputModification {
    #[derivative(Default)]
    OneForOne,

    SingleValue,

    OnePerSubgroup,

    /// Make the output buffer be oversized to compare how much the final
    /// accumulation costs
    FixedSize(DeviceSize),
}

#[derive(Derivative)]
#[derivative(Default, Clone)]
pub struct ComputeParameters {
    #[derivative(Default(value = "1"))]
    pub vectorization_factor: u32,

    pub clear_buffer: bool,

    pub output: OutputModification,

    pub skip_cpu_final_accumulation: bool,
}

impl<Type> ComputeExecuteUtil<Type>
where
    Type: Copy + NumCast + Pod + BufferContents + PartialEq + Zero + Debug + Sum,
{
    #[inline(always)]
    fn generic_setup<SC, Acc, INIT>(
        vulkan: &mut VulkanData,
        cs: &ShaderModule,
        sc: SC,
        parameters: ComputeParameters,

        accumulate: Acc,
        specialized_init: INIT,
    ) -> Self
    where
        SC: SpecializationConstants,
        Acc: 'static + Fn(Type, Type) -> Type,
        INIT: FnOnce(
            &mut VulkanData,
            &Arc<ComputePipeline>,
        ) -> (Vector2<u32>, Arc<PersistentDescriptorSet>, Type),
    {
        let pipeline = ComputePipeline::new(
            vulkan.device.clone(),
            cs.entry_point("main").unwrap(),
            &sc,
            None,
            |_| {},
        )
        .unwrap();

        let (viewport_size, set, expected_result) = specialized_init(vulkan, &pipeline);

        Self {
            viewport_size,
            pipeline,
            set,
            instance_id: 1,
            expected_result,
            parameters,

            accumulate: Box::new(accumulate),
            t: Default::default(),
        }
    }

    #[inline(always)]
    pub fn setup_storage_buffer<SC, Acc>(
        vulkan: &mut VulkanData,
        data_size: Vector2<u32>,
        fs: &ShaderModule,
        sc: SC,

        parameters: ComputeParameters,

        accumulate: Acc,
    ) -> Self
    where
        SC: SpecializationConstants,
        Acc: 'static + Fn(Type, Type) -> Type,
    {
        let total = data_size.x * data_size.y;
        let gen_data = generate_data(total).collect_vec();
        let expected = gen_data.iter().copied().reduce(&accumulate).unwrap();

        let mut executor = Self::generic_setup(
            vulkan,
            fs,
            sc,
            parameters.clone(),
            accumulate,
            move |vulkan, pipeline| {
                let mut command_buffer = vulkan.create_command_buffer();
                let data: Subbuffer<[Type]> =
                    vulkan.create_storage_buffer(&mut command_buffer, gen_data.iter().copied());

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

                (Vector2::new(data_size.x, 1), set, expected)
            },
        );

        assert_eq!(
            executor.viewport_size.x % 64,
            0,
            "Dimension X must be a multiple of 64 because of workgroup sizes"
        );

        assert_eq!(
            data_size.y % parameters.vectorization_factor,
            0,
            "Dimension Y must be a multiple of the vectorization factor"
        );
        executor.instance_id = data_size.y / parameters.vectorization_factor;

        executor
    }

    #[inline(always)]
    pub fn run(&mut self, vulkan: &mut VulkanData, separate_read_buffer: bool) {
        let mut command_buffer = vulkan.create_command_buffer();

        let target: Subbuffer<[Type]> = Buffer::new_slice(
            &vulkan.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
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
            match self.parameters.output {
                OutputModification::OneForOne => {
                    (self.viewport_size.x * self.viewport_size.y) as DeviceSize
                },
                OutputModification::SingleValue => 1,
                OutputModification::OnePerSubgroup => {
                    ((self.viewport_size.x * self.viewport_size.y) as DeviceSize)
                        .div_ceil(vulkan.physical_device.properties().subgroup_size.unwrap()
                            as DeviceSize)
                },
                OutputModification::FixedSize(size) => {
                    size.max((self.viewport_size.x * self.viewport_size.y) as DeviceSize)
                },
            } * (self.parameters.vectorization_factor as DeviceSize),
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

        if self.parameters.clear_buffer {
            command_buffer
                .fill_buffer(target.clone().into_bytes().cast_aligned(), 0)
                .unwrap();
        }

        let target_set = PersistentDescriptorSet::new(
            &vulkan.descriptor_set_allocator,
            self.pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, target.clone())],
        )
        .unwrap();

        assert_eq!(self.viewport_size.x % 64, 0);
        command_buffer
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                self.set.clone(),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                1,
                target_set,
            )
            .push_constants(self.pipeline.layout().clone(), 0, self.instance_id)
            .dispatch([self.viewport_size.x / 64, 1, 1])
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

        // println!("\n\n\n{:x?}\n", &read_buffer.read().unwrap() as &[_]);

        if !self.parameters.skip_cpu_final_accumulation {
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
        }
    }
}
