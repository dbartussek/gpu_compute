#![feature(int_roundings)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{ExecuteUtil, OutputKind},
    execute_util_compute::ComputeExecuteUtil,
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("gpu_sum");
    g.measurement_time(Duration::from_secs(30));

    for y in [1_000_000_000 / 256, 1, 100]
        .into_iter()
        .chain((0..=15_000).step_by(5000).filter(|v| *v != 0))
        .chain([15_625])
    {
        g.bench_with_input(
            BenchmarkId::new("buffer_to_rendertarget", y * 256),
            &y,
            |b, y| {
                let shader = attach_discard_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ExecuteUtil::setup_storage_buffer(
                    &mut vulkan,
                    Vector2::new(256, *y),
                    &shader,
                    attach_discard_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: 256,
                        TEXTURE_SIZE_Y: 1,
                    },
                    OutputKind::Attachment,
                    1,
                );

                b.iter(|| {
                    execute.run(&mut vulkan);
                });
            },
        );
        g.bench_with_input(BenchmarkId::new("buffer_to_buffer", y * 256), &y, |b, y| {
            let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
            let mut execute = ExecuteUtil::setup_storage_buffer(
                &mut vulkan,
                Vector2::new(256 * 16, y.div_ceil(16)),
                &shader,
                buffer_none_sbuffer_loop::SpecializationConstants {
                    TEXTURE_SIZE_X: 256 * 16,
                    TEXTURE_SIZE_Y: 1,
                },
                OutputKind::Buffer,
                1,
            );

            b.iter(|| {
                execute.run(&mut vulkan);
            });
        });
        continue;
        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer", y * 256),
            &y,
            |b, y| {
                let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::setup_storage_buffer(
                    &mut vulkan,
                    Vector2::new(256, *y),
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: 256,
                        TEXTURE_SIZE_Y: 1,
                    },
                );

                b.iter(|| {
                    execute.run(&mut vulkan);
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
