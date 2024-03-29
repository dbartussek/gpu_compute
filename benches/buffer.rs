use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{ExecuteParameters, ExecuteUtil, OutputKind, QuadMethod},
    execute_util_compute::{ComputeExecuteUtil, ComputeParameters, OutputModification},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("gpu_sum");
    // g.measurement_time(std::time::Duration::from_secs(30));
    g.sample_size(10);

    let profiling_sizes = vulkan.profiling_sizes();


    println!("{:X?}", profiling_sizes);
    for y in profiling_sizes.clone() {
        let data_size = Vector2::new(vulkan.gpu_thread_count(), y / vulkan.gpu_thread_count());

        for method in QuadMethod::all(&vulkan).iter().copied() {
            for framebuffer_y in [1, 2, 32, 64] {
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
                                TEXTURE_SIZE_Y: framebuffer_y as _,
                            },
                            ExecuteParameters {
                                output: OutputKind::Attachment,
                                quad_method: method,
                                framebuffer_y,
                                ..Default::default()
                            },
                            |a, b| a + b,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, true);
                        });
                    },
                );

                if data_size.y <= 32768 {
                    g.bench_with_input(
                        BenchmarkId::new(format!("sampler2d_to_rendertarget_{suffix}"), y),
                        &y,
                        |b, _| {
                            let shader =
                                attach_none_sampled_loop::load(vulkan.device.clone()).unwrap();
                            let mut execute = ExecuteUtil::<u32>::setup_2d_sampler(
                                &mut vulkan,
                                data_size,
                                &shader,
                                attach_none_sampled_loop::SpecializationConstants {
                                    TEXTURE_SIZE_X: (data_size.x / framebuffer_y) as _,
                                    TEXTURE_SIZE_Y: framebuffer_y as _,
                                },
                                ExecuteParameters {
                                    output: OutputKind::Attachment,
                                    quad_method: method,
                                    framebuffer_y,
                                    ..Default::default()
                                },
                                |a, b| a + b,
                            );

                            b.iter(|| {
                                execute.run(&mut vulkan, true);
                            });
                        },
                    );
                }

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
                                TEXTURE_SIZE_Y: framebuffer_y as _,
                            },
                            ExecuteParameters {
                                output: OutputKind::Buffer,
                                quad_method: method,
                                framebuffer_y,
                                ..Default::default()
                            },
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
                                TEXTURE_SIZE_Y: framebuffer_y as _,
                            },
                            ExecuteParameters {
                                output: OutputKind::Buffer,
                                quad_method: method,
                                framebuffer_y,
                                ..Default::default()
                            },
                            |a, b| a + b,
                        );

                        b.iter(|| {
                            execute.run(&mut vulkan, false);
                        });
                    },
                );

                if data_size.y % 4 == 0 {
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
                                    TEXTURE_SIZE_Y: framebuffer_y as _,
                                },
                                ExecuteParameters {
                                    output: OutputKind::Buffer,
                                    quad_method: method,
                                    framebuffer_y,
                                    vectorization_factor: 4,
                                    ..Default::default()
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
        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer_subgroup_cpu_visible_memory", y),
            &y,
            |b, _| {
                let shader = compute_none_groupbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_groupbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::OnePerSubgroup,
                        ..Default::default()
                    },
                    |a, b| a + b,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new(
                "compute_buffer_to_buffer_subgroup_decimate_cpu_visible_memory",
                y,
            ),
            &y,
            |b, _| {
                let shader =
                    compute_none_groupbuffer_decimate_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_groupbuffer_decimate_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::OnePerSubgroup,
                        ..Default::default()
                    },
                    |a, b| a + b,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new(
                "compute_buffer_to_buffer_atomic_subgroup_cpu_visible_memory",
                y,
            ),
            &y,
            |b, _| {
                let shader =
                    compute_none_subgroup_abuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_subgroup_abuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::SingleValue,
                        clear_buffer: true,
                        ..Default::default()
                    },
                    |a, b| a + b,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new("compute_none_subgroup_casbuffer_loop", y),
            &y,
            |b, _| {
                let shader =
                    compute_none_subgroup_casbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_subgroup_casbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::SingleValue,
                        clear_buffer: true,
                        ..Default::default()
                    },
                    |a, b| a + b,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new(
                "compute_buffer_to_buffer_atomic_subgroup_cpu_visible_memory_65536_threads",
                y,
            ),
            &y,
            |b, _| {
                let shader =
                    compute_none_subgroup_abuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_subgroup_abuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::SingleValue,
                        clear_buffer: true,
                        override_thread_count: Some(65536),
                        ..Default::default()
                    },
                    |a, b| a + b,
                );

                b.iter(|| {
                    execute.run(&mut vulkan, false);
                });
            },
        );

        g.bench_with_input(
            BenchmarkId::new("compute_buffer_to_buffer_atomic_add_cpu_visible_memory", y),
            &y,
            |b, _| {
                let shader =
                    compute_none_atomic_add_buffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_atomic_add_buffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::SingleValue,
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
        g.bench_with_input(
            BenchmarkId::new(
                "compute_buffer_to_buffer_atomic_add_cpu_visible_memory_65536_threads",
                y,
            ),
            &y,
            |b, _| {
                let shader =
                    compute_none_atomic_add_buffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    compute_none_atomic_add_buffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ComputeParameters {
                        output: OutputModification::SingleValue,
                        clear_buffer: true,
                        override_thread_count: Some(65536),
                        ..ComputeParameters::default()
                    },
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
        let y = profiling_sizes.first().copied().unwrap();
        let data_size = Vector2::new(
            vulkan.gpu_thread_count() / 2,
            y * 2 / vulkan.gpu_thread_count(),
        );

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
                        output: OutputModification::SingleValue,
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
        spirv_version: "1.3",
    }
}
mod compute_none_subgroup_casbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_subgroup_casbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
        spirv_version: "1.3",
    }
}
mod compute_none_atomic_add_buffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_atomic_add_buffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}
mod compute_none_subgroup_abuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_subgroup_abuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
        spirv_version: "1.3",
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

mod compute_none_groupbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_groupbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
        spirv_version: "1.3",
    }
}

mod compute_none_groupbuffer_decimate_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_groupbuffer_decimate_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
        spirv_version: "1.3",
    }
}
