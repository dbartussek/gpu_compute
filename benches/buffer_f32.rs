use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::{
    execute_util::{BlendMethod, ExecuteParameters, ExecuteUtil, OutputKind, QuadMethod},
    execute_util_compute::{ComputeExecuteUtil, ComputeParameters},
    vulkan_util::VulkanData,
};
use itertools::Itertools;
use nalgebra::Vector2;
use num::Float;
use vulkano::format::{ClearValue, Format};

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("gpu_min_f32");
    // g.measurement_time(std::time::Duration::from_secs(30));
    g.sample_size(10);

    let profiling_sizes = vulkan.profiling_sizes();


    println!("normal sizes: {:X?}", profiling_sizes);
    println!("normal sizes: {:?}", profiling_sizes);
    println!(
        "vector sizes: {:X?}",
        profiling_sizes
            .iter()
            .copied()
            .filter(|v| v % 4 == 0)
            .collect_vec()
    );
    println!(
        "vector sizes: {:?}",
        profiling_sizes
            .iter()
            .copied()
            .filter(|v| v % 4 == 0)
            .collect_vec()
    );
    for y in profiling_sizes.clone() {
        let data_size = Vector2::new(vulkan.gpu_thread_count(), y / vulkan.gpu_thread_count());

        g.bench_with_input(
            BenchmarkId::new("graphics_buffer_to_buffer", y),
            &y,
            |b, _| {
                let shader = none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ExecuteUtil::<f32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ExecuteParameters {
                        output: OutputKind::Buffer,
                        quad_method: QuadMethod::large_triangle,
                        ..Default::default()
                    },
                    |a, b| a.min(b),
                );

                b.iter(|| {
                    execute.run(&mut vulkan, true);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new("graphics_buffer_to_rendertarget", y),
            &y,
            |b, _| {
                let shader = attach_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
                let mut execute = ExecuteUtil::<f32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    attach_none_sbuffer_loop::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ExecuteParameters {
                        output: OutputKind::RenderAttachment(Format::R32_SFLOAT),
                        quad_method: QuadMethod::large_triangle,
                        clear_value: ClearValue::Float([f32::infinity(); 4]),
                        ..Default::default()
                    },
                    |a, b| a.min(b),
                );

                b.iter(|| {
                    execute.run(&mut vulkan, true);
                });
            },
        );
        g.bench_with_input(
            BenchmarkId::new("graphics_buffer_to_rendertarget_blend", y),
            &y,
            |b, _| {
                let shader = attach_none_sbuffer_many::load(vulkan.device.clone()).unwrap();
                let mut execute = ExecuteUtil::<f32>::setup_storage_buffer(
                    &mut vulkan,
                    data_size,
                    &shader,
                    attach_none_sbuffer_many::SpecializationConstants {
                        TEXTURE_SIZE_X: data_size.x as _,
                        TEXTURE_SIZE_Y: 1,
                    },
                    ExecuteParameters {
                        output: OutputKind::RenderAttachment(Format::R32_SFLOAT),
                        quad_method: QuadMethod::large_triangle,
                        clear_value: ClearValue::Float([f32::infinity(); 4]),
                        blend: Some(BlendMethod::Min),
                        use_instances_and_blend: true,
                        ..Default::default()
                    },
                    |a, b| a.min(b),
                );

                b.iter(|| {
                    execute.run(&mut vulkan, true);
                });
            },
        );

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

mod none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_min/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("DATA_TYPE", "float")],
    }
}
mod attach_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_min/attach_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("DATA_TYPE", "float")],
    }
}
mod attach_none_sbuffer_many {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_min/attach_none_sbuffer_many.glsl",
        include: ["shaders/pluggable"],
        define: [("DATA_TYPE", "float")],
    }
}

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
