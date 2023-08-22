use criterion::{criterion_group, criterion_main, Criterion};
use gpu_compute::{
    an_external_function, do_virtual_call,
    execute_util::{ExecuteParameters, ExecuteUtil},
    execute_util_compute::{ComputeExecuteUtil, ComputeParameters},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;
use std::{ffi::c_int, hint::black_box, time::Duration};
use vulkano::{command_buffer::PrimaryCommandBufferAbstract, sync::GpuFuture};

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("call_times");
    g.measurement_time(Duration::from_secs(30));
    // g.sample_size(1000);

    g.bench_function("an_external_function", |b| {
        b.iter(|| unsafe { an_external_function() })
    });
    g.bench_function("virtual_function", |b| {
        extern "C" fn virtual_called() -> c_int {
            0
        }

        b.iter(|| unsafe { do_virtual_call(black_box(virtual_called)) })
    });

    g.bench_function("submit_and_wait", |b| {
        b.iter(|| {
            vulkan
                .create_command_buffer()
                .build()
                .unwrap()
                .execute(vulkan.queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
        })
    });

    g.bench_function("run_shader", |b| {
        let shader = none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
        let mut execute = ExecuteUtil::<u32>::setup_storage_buffer(
            &mut vulkan,
            Vector2::new(1, 1),
            &shader,
            none_sbuffer_loop::SpecializationConstants {
                TEXTURE_SIZE_X: 1,
                TEXTURE_SIZE_Y: 1,
            },
            ExecuteParameters {
                ..Default::default()
            },
            |a, b| a + b,
        );

        b.iter(|| {
            execute.run(&mut vulkan, true);
        });
    });

    g.bench_function("run_compute_shader", |b| {
        let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
        let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
            &mut vulkan,
            Vector2::new(64, 1),
            &shader,
            compute_none_sbuffer_loop::SpecializationConstants {
                TEXTURE_SIZE_X: 1,
                TEXTURE_SIZE_Y: 1,
            },
            ComputeParameters {
                ..ComputeParameters::default()
            },
            |a, b| a + b,
        );

        b.iter(|| {
            execute.run(&mut vulkan, true);
        });
    });

    #[cfg(feature = "cuda")]
    g.bench_function("cuda_empty_kernel", |b| {
        b.iter(|| unsafe {gpu_compute::cuda_empty_kernel()})
    });

    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);


mod none_sbuffer_loop {
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
