#![feature(int_roundings)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{ExecuteUtil, OutputKind},
    execute_util_compute::ComputeExecuteUtil,
    vulkan_util::VulkanData,
};
use itertools::Itertools;
use nalgebra::Vector2;
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("gpu_sum");
    g.measurement_time(Duration::from_secs(30));

    for y in [1u32]
        .into_iter()
        .chain((0..=(50_000 * 4)).step_by(256 * 32 * 4))
        .chain(((50_000 * 4)..=(50_000 * 64)).step_by(256 * 256 * 4))

        // .chain(((0)..=(50_000 * 64* 64)).step_by(256 * 256 * 4 * 64))
        .filter(|v| *v != 0)
        .map(|v| v.div_ceil(256 * 32) * 256 * 32)
        .unique()
    {
        let data_size = Vector2::new(256 * 32, y.div_ceil(32 * 256));

        g.bench_with_input(BenchmarkId::new("buffer_to_rendertarget", y), &y, |b, _| {
            let shader = attach_discard_sbuffer_loop::load(vulkan.device.clone()).unwrap();
            let mut execute = ExecuteUtil::setup_storage_buffer(
                &mut vulkan,
                data_size,
                &shader,
                attach_discard_sbuffer_loop::SpecializationConstants {
                    TEXTURE_SIZE_X: data_size.x as _,
                    TEXTURE_SIZE_Y: 1,
                },
                OutputKind::Attachment,
                1,
                1,
            );

            b.iter(|| {
                execute.run(&mut vulkan, true);
            });
        });

        g.bench_with_input(BenchmarkId::new("buffer_to_buffer", y), &y, |b, _| {
            let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
            let mut execute = ExecuteUtil::setup_storage_buffer(
                &mut vulkan,
                data_size,
                &shader,
                buffer_none_sbuffer_loop::SpecializationConstants {
                    TEXTURE_SIZE_X: data_size.x as _,
                    TEXTURE_SIZE_Y: 1,
                },
                OutputKind::Buffer,
                1,
                1,
            );

            b.iter(|| {
                execute.run(&mut vulkan, true);
            });
        });
        g.bench_with_input(BenchmarkId::new("buffer_to_buffer_cpu_visible_memory", y), &y, |b, _| {
            let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
            let mut execute = ExecuteUtil::setup_storage_buffer(
                &mut vulkan,
                data_size,
                &shader,
                buffer_none_sbuffer_loop::SpecializationConstants {
                    TEXTURE_SIZE_X: data_size.x as _,
                    TEXTURE_SIZE_Y: 1,
                },
                OutputKind::Buffer,
                1,
                1,
            );

            b.iter(|| {
                execute.run(&mut vulkan, false);
            });
        });

        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer", y),
            &y,
            |b, _| {
                let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    1,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, true);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer_cpu_visible_memory", y),
            &y,
            |b, _| {
                let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    1,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );
    }

    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

mod attach_discard_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/attach_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}

mod buffer_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}

mod compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}
