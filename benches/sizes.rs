#![feature(int_roundings)]

use std::collections::HashSet;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{execute_util::{ExecuteUtil, OutputKind, QuadMethod}, vulkan_util::VulkanData};
use nalgebra::Vector2;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("optimal_accumulate_size");
    // g.measurement_time(std::time::Duration::from_secs(30));
    g.sample_size(10);

    let mut dedup = HashSet::new();

    for data_size in vulkan.profiling_sizes().iter().copied()
    {
        for group_size in [
            64,
            256,
            512,
            4096,
            8192,
            16384,
            32768,
            65536,
            // 8072,
            // 8072 * 2,
            // 8072 * 4,
        ] {
            let data_size = data_size.div_ceil(group_size) * group_size;

            if !dedup.insert((group_size, data_size)) {
                continue;
            }

            g.bench_with_input(
                BenchmarkId::new(format!("group-{}", group_size), data_size),
                &data_size,
                |b, data_size| {
                    let frame_y = 4;

                    let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                    let mut execute = ExecuteUtil::<u32>::setup_storage_buffer(
                        &mut vulkan,
                        Vector2::new(group_size, data_size.div_ceil(group_size)),
                        &shader,
                        buffer_none_sbuffer_loop::SpecializationConstants {
                            TEXTURE_SIZE_X: (group_size as i32) / frame_y,
                            TEXTURE_SIZE_Y: frame_y,
                        },
                        OutputKind::Buffer,
                        QuadMethod::large_triangle,
                        frame_y as _,
                        1,
                        |a, b| a + b,
                    );

                    b.iter(|| {
                        execute.run(&mut vulkan, true);
                    });
                },
            );
        }
    }

    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

mod buffer_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_sum/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}
