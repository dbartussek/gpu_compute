#![feature(int_roundings)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{ExecuteUtil, OutputKind},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("optimal_accumulate_size");
    g.measurement_time(Duration::from_secs(10));

    for data_size in [1u32 << 16, 1<<17, 1<<27, 1<<28, 1<<29] //(16..30).map(|shift| 1u32 << shift)
    {
        for group_size in [256, 512, 4096, 8192, 16384, 32768, 65536, 8072, 8072*2,8072*4] {
            g.bench_with_input(
                BenchmarkId::new(format!("group-{}", group_size), data_size),
                &data_size,
                |b, data_size| {
                    let frame_y = 4;

                    let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                    let mut execute = ExecuteUtil::setup_storage_buffer(
                        &mut vulkan,
                        Vector2::new(group_size, data_size.div_ceil(group_size)),
                        &shader,
                        buffer_none_sbuffer_loop::SpecializationConstants {
                            TEXTURE_SIZE_X: (group_size as i32) / frame_y,
                            TEXTURE_SIZE_Y: frame_y,
                        },
                        OutputKind::Buffer,
                        frame_y as _,
                    );

                    b.iter(|| {
                        execute.run(&mut vulkan);
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
        path: "shaders/instances/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}
