use criterion::{criterion_group, criterion_main, Criterion};
use gpu_compute::{
    an_external_function, do_virtual_call,
    execute_util::{ExecuteUtil, OutputKind},
    vulkan_util::VulkanData,
};
use std::{ffi::c_int, hint::black_box};
use vulkano::{command_buffer::PrimaryCommandBufferAbstract, sync::GpuFuture};

fn criterion_benchmark(c: &mut Criterion) {
    let mut vulkan = VulkanData::init();

    let mut g = c.benchmark_group("call_times");

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
        let shader = attach_discard_sampled_many::load(vulkan.device.clone()).unwrap();
        let mut execute = ExecuteUtil::setup_1d_sampler(
            &mut vulkan,
            1,
            &shader,
            attach_discard_sampled_many::SpecializationConstants {
                TEXTURE_SIZE_X: 1,
                TEXTURE_SIZE_Y: 1,
            },
            OutputKind::Attachment,
        );

        b.iter(|| {
            execute.run(&mut vulkan, true);
        });
    });

    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);


mod attach_discard_sampled_many {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/attach_discard_sampled1D_many.glsl",
        include: ["shaders/pluggable"],
    }
}
