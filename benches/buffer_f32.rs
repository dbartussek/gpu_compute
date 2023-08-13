use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util_compute::{ComputeExecuteUtil, ComputeParameters},
    vulkan_util::VulkanData,
    GPU_THREAD_COUNT, PROFILING_SIZES,
};
use itertools::Itertools;
use nalgebra::Vector2;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("gpu_min_f32");
    // g.measurement_time(std::time::Duration::from_secs(30));
    g.sample_size(10);


    println!("normal sizes: {:X?}", *PROFILING_SIZES);
    println!("normal sizes: {:?}", *PROFILING_SIZES);
    println!(
        "vector sizes: {:X?}",
        PROFILING_SIZES
            .iter()
            .copied()
            .filter(|v| v % 4 == 0)
            .collect_vec()
    );
    println!(
        "vector sizes: {:?}",
        PROFILING_SIZES
            .iter()
            .copied()
            .filter(|v| v % 4 == 0)
            .collect_vec()
    );
    for y in PROFILING_SIZES.clone() {
        let data_size = Vector2::new(GPU_THREAD_COUNT, y / GPU_THREAD_COUNT);

        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer", y),
            &y,
            |b, _| {
                let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<f32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters::default(),
                    |a, b| a.min(b),
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
                let mut execute = ComputeExecuteUtil::<f32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters::default(),
                    |a, b| a.min(b),
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );


        if data_size.y % 4 == 0 {
            g.bench_with_input(
                BenchmarkId::new("vector_compute_buffer_to_buffer", y),
                &y,
                |b, _| {
                    let shader =
                        vector_compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                    let mut execute = ComputeExecuteUtil::<f32>::setup_storage_buffer(
                        &mut vulkan,
                        data_size,
                        &shader,
                        vector_compute_none_sbuffer_loop::SpecializationConstants {
                            TEXTURE_SIZE_X: data_size.x as _,
                            TEXTURE_SIZE_Y: 1,
                        },
                        ComputeParameters {
                            vectorization_factor: 4,
                            ..ComputeParameters::default()
                        },
                        |a, b| a.min(b),
                    );

                    b.iter(|| {
                        execute.run(&mut vulkan, true);
                    });
                },
            );
            g.bench_with_input(
                BenchmarkId::new("vector_compute_buffer_to_buffer_visible_memory", y),
                &y,
                |b, _| {
                    let shader =
                        vector_compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                    let mut execute = ComputeExecuteUtil::<f32>::setup_storage_buffer(
                        &mut vulkan,
                        data_size,
                        &shader,
                        vector_compute_none_sbuffer_loop::SpecializationConstants {
                            TEXTURE_SIZE_X: data_size.x as _,
                            TEXTURE_SIZE_Y: 1,
                        },
                        ComputeParameters {
                            vectorization_factor: 4,
                            ..ComputeParameters::default()
                        },
                        |a, b| a.min(b),
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

mod compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_min/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1"), ("DATA_TYPE", "float")],
    }
}
mod vector_compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_min/vectorized/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1"), ("DATA_TYPE", "vec4")],
    }
}
