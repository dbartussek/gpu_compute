use crate::{
    execute_util::{generate_data, s32},
    vulkan_util::VulkanData,
};
use itertools::Itertools;
use nalgebra::Vector2;
use std::{hint::black_box, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{CopyBufferInfo, PrimaryCommandBufferAbstract},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{ShaderModule, SpecializationConstants},
    sync::GpuFuture,
    DeviceSize,
};

pub struct ComputeExecuteUtil {
    viewport_size: Vector2<u32>,
    vectorization_factor: u32,

    pipeline: Arc<ComputePipeline>,
    set: Arc<PersistentDescriptorSet>,

    instance_id: u32,

    expected_result: u32,
}

impl ComputeExecuteUtil {
    #[inline(always)]
    fn generic_setup<SC, INIT>(
        vulkan: &mut VulkanData,
        cs: &ShaderModule,
        sc: SC,
        vectorization_factor: u32,

        specialized_init: INIT,
    ) -> Self
    where
        SC: SpecializationConstants,
        INIT: FnOnce(
            &mut VulkanData,
            &Arc<ComputePipeline>,
        ) -> (Vector2<u32>, Arc<PersistentDescriptorSet>, u32),
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
            vectorization_factor,
        }
    }

    #[inline(always)]
    pub fn setup_storage_buffer<SC>(
        vulkan: &mut VulkanData,
        data_size: Vector2<u32>,
        fs: &ShaderModule,
        sc: SC,

        vectorization_factor: u32,
    ) -> Self
    where
        SC: SpecializationConstants,
    {
        let mut executor =
            Self::generic_setup(vulkan, fs, sc, vectorization_factor, |vulkan, pipeline| {
                let total = data_size.x * data_size.y;
                let gen_data = generate_data(total).collect_vec();

                let mut command_buffer = vulkan.create_command_buffer();
                let data =
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

                (Vector2::new(data_size.x, 1), set, s32(gen_data))
            });

        assert_eq!(
            executor.viewport_size.x % 64,
            0,
            "Dimension X must be a multiple of 64 because of workgroup sizes"
        );

        assert_eq!(
            data_size.y % vectorization_factor,
            0,
            "Dimension Y must be a multiple of the vectorization factor"
        );
        executor.instance_id = data_size.y / vectorization_factor;

        executor
    }

    #[inline(always)]
    pub fn run(&mut self, vulkan: &mut VulkanData, separate_read_buffer: bool) {
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
            (self.viewport_size.x * self.viewport_size.y * self.vectorization_factor) as DeviceSize,
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

        // dbg!(&read_buffer.read().unwrap() as &[_]);

        let result = black_box(s32(read_buffer.read().unwrap().iter().copied()));
        assert_eq!(result, self.expected_result);
    }
}
