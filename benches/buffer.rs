use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{ExecuteUtil, OutputKind, QuadMethod},
    execute_util_compute::{ComputeExecuteUtil, ComputeParameters},
    vulkan_util::VulkanData,
    GPU_THREAD_COUNT, PROFILING_SIZES,
};
use nalgebra::Vector2;
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("gpu_sum");
    g.measurement_time(Duration::from_secs(30));


    println!("{:X?}", *PROFILING_SIZES);
    for y in PROFILING_SIZES.clone() {
        let data_size = Vector2::new(GPU_THREAD_COUNT, y / GPU_THREAD_COUNT);

        for method in QuadMethod::all(&vulkan).iter().copied() {
            for framebuffer_y in [1, 2, 32] {
                let suffix = format!(
                    "{method:?}_{framebuffer_x}x{framebuffer_y}",
                    framebuffer_x = data_size.x / framebuffer_y
                );

                g.bench_with_input(
                    BenchmarkId::new(format!("buffer_to_rendertarget_{suffix}"), y),
                    &y,
                    |b, _| {
                        let shader =
                            attach_discard_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                        let mut execute = ExecuteUtil::<u32>::setup_storage_buffer(
                            &mut vulkan,
                            data_size,
                            &shader,
                            attach_discard_sbuffer_loop::SpecializationConstants {
                                TEXTURE_SIZE_X: (data_size.x / framebuffer_y) as _,
                                TEXTURE_SIZE_Y: data_size.y as _,
                            },
                            OutputKind::Attachment,
                            method,
                            framebuffer_y,
                            1,
                            |a, b| a + b,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, true);
                        });
                    },
                );

                g.bench_with_input(
                    BenchmarkId::new(format!("sampler2d_to_rendertarget_{suffix}"), y),
                    &y,
                    |b, _| {
                        let shader = attach_none_sampled_loop::load(vulkan.device.clone()).unwrap();
                        let mut execute = ExecuteUtil::<u32>::setup_2d_sampler(
                            &mut vulkan,
                            data_size,
                            &shader,
                            attach_none_sampled_loop::SpecializationConstants {
                                TEXTURE_SIZE_X: (data_size.x / framebuffer_y) as _,
                                TEXTURE_SIZE_Y: data_size.y as _,
                            },
                            OutputKind::Attachment,
                            method,
                            framebuffer_y,
                            |a, b| a + b,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, true);
                        });
                    },
                );

                g.bench_with_input(
                    BenchmarkId::new(format!("buffer_to_buffer_{suffix}"), y),
                    &y,
                    |b, _| {
                        let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                        let mut execute = ExecuteUtil::<u32>::setup_storage_buffer(
                            &mut vulkan,
                            data_size,
                            &shader,
                            buffer_none_sbuffer_loop::SpecializationConstants {
                                TEXTURE_SIZE_X: (data_size.x / framebuffer_y) as _,
                                TEXTURE_SIZE_Y: data_size.y as _,
                            },
                            OutputKind::Buffer,
                            method,
                            framebuffer_y,
                            1,
                            |a, b| a + b,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, true);
                        });
                    },
                );
                g.bench_with_input(
                    BenchmarkId::new(format!("buffer_to_buffer_cpu_visible_memory_{suffix}"), y),
                    &y,
                    |b, _| {
                        let shader = buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                        let mut execute = ExecuteUtil::<u32>::setup_storage_buffer(
                            &mut vulkan,
                            data_size,
                            &shader,
                            buffer_none_sbuffer_loop::SpecializationConstants {
                                TEXTURE_SIZE_X: (data_size.x / framebuffer_y) as _,
                                TEXTURE_SIZE_Y: data_size.y as _,
                            },
                            OutputKind::Buffer,
                            method,
                            framebuffer_y,
                            1,
                            |a, b| a + b,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, false);
                        });
                    },
                );

                if false {
                    g.bench_with_input(
                        BenchmarkId::new(format!("vector_buffer_to_buffer_{suffix}"), y),
                        &y,
                        |b, _| {
                            let shader =
                                vector_buffer_none_sbuffer_loop::load(vulkan.device.clone())
                                    .unwrap();
                            let mut execute = ExecuteUtil::<u32>::setup_storage_buffer(
                                &mut vulkan,
                                data_size,
                                &shader,
                                vector_buffer_none_sbuffer_loop::SpecializationConstants {
                                    TEXTURE_SIZE_X: (data_size.x / framebuffer_y) as _,
                                    TEXTURE_SIZE_Y: data_size.y as _,
                                },
                                OutputKind::Buffer,
                                method,
                                framebuffer_y,
                                4,
                                |a, b| a + b,
                            );

                            b.iter(|| {
                                execute.run(&mut vulkan, true);
                            });
                        },
                    );
                }
            }
        }

        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer", y),
            &y,
            |b, _| {
                let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters::default(),
                    |a, b| a + b,
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
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters::default(),
                    |a, b| a + b,
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
                    let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
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
                        |a, b| a + b,
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
                    let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
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
                        |a, b| a + b,
                    );

                    b.iter(|| {
                        execute.run(&mut vulkan, true);
                    });
                },
            );
        }
    }

    {
        // This has AWFUL performance
        let y = PROFILING_SIZES.first().copied().unwrap();
        let data_size = Vector2::new(GPU_THREAD_COUNT, y / GPU_THREAD_COUNT);

        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer_atomic_cpu_visible_memory", y),
            &y,
            |b, _| {
                let shader = compute_none_abuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_abuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        single_output_value: true,
                        clear_buffer: true,
                        ..ComputeParameters::default()
                    },
                    |a, b| a + b,
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
        path: "shaders/instances/gpu_sum/attach_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}

mod buffer_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_sum/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}

mod compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}
mod compute_none_abuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_abuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}

mod vector_buffer_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_sum/vectorized/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}

mod vector_compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/vectorized/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}


mod attach_none_sampled_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_sum/attach_none_sampled2D_loop.glsl",
        include: ["shaders/pluggable"],
    }
}
