use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{ExecuteUtil, OutputKind},
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
        g.bench_with_input(
            BenchmarkId::new("buffer_to_buffer_cpu_visible_memory", y),
            &y,
            |b, _| {
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
            },
        );

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
                    ComputeParameters::default(),
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
                    ComputeParameters::default(),
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );


        if data_size.y % 4 == 0 {
            if false {
                g.bench_with_input(
                    BenchmarkId::new("vector_buffer_to_buffer", y),
                    &y,
                    |b, _| {
                        let shader =
                            vector_buffer_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                        let mut execute = ExecuteUtil::setup_storage_buffer(
                            &mut vulkan,
                            data_size,
                            &shader,
                            vector_buffer_none_sbuffer_loop::SpecializationConstants {
                                TEXTURE_SIZE_X: data_size.x as _,
                                TEXTURE_SIZE_Y: 1,
                            },
                            OutputKind::Buffer,
                            1,
                            4,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, true);
                        });
                    },
                );
            }
            g.bench_with_input(
                BenchmarkId::new("vector_compute_buffer_to_buffer", y),
                &y,
                |b, _| {
                    let shader =
                        vector_compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                    let mut execute = ComputeExecuteUtil::setup_storage_buffer(
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
                    let mut execute = ComputeExecuteUtil::setup_storage_buffer(
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
                let mut execute = ComputeExecuteUtil::setup_storage_buffer(
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
mod compute_none_abuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/buffer_none_abuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}

mod vector_buffer_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/vectorized/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
    }
}

mod vector_compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/vectorized/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}
