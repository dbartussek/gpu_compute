#![feature(int_roundings)]

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use gpu_compute::{
    execute_util::{generate_data, s32},
    GPU_THREAD_COUNT,
};
use itertools::Itertools;
use ocl::{r#async::BufferSink, Buffer, MemFlags, ProQue, WriteGuard};
use ocl_futures::future::Future;

fn do_cl_bench(g: &mut BenchmarkGroup<WallTime>, data_size: u32, src: &str, name: &str) {
    g.bench_with_input(BenchmarkId::new(name, data_size), &data_size, |b, _| {
        let cl_program = ProQue::builder()
            .src(src)
            .dims(GPU_THREAD_COUNT)
            .build()
            .unwrap();

        let source_buffer = Buffer::<u32>::builder()
            .queue(cl_program.queue().clone())
            .flags(
                MemFlags::empty()
                    .read_only()
                    .host_write_only()
                    .alloc_host_ptr(),
            )
            .len(data_size)
            .build()
            .unwrap();

        {
            let source_sink = unsafe {
                BufferSink::from_buffer(
                    source_buffer.clone(),
                    Some(cl_program.queue().clone()),
                    0,
                    data_size as _,
                )
            }
            .unwrap();
            let writer = source_sink.write();
            let mut writer = writer.wait().unwrap();
            writer.copy_from_slice(&generate_data(data_size).collect_vec());

            let source_sink: BufferSink<u32> = WriteGuard::release(writer).into();
            source_sink.flush().enq().unwrap().wait().unwrap();
        }

        let output_buffer = cl_program
            .buffer_builder::<u32>()
            .flags(MemFlags::empty().host_read_only().write_only())
            .build()
            .unwrap();

        let kernel = cl_program
            .kernel_builder("sum")
            .arg(GPU_THREAD_COUNT as u64)
            .arg((data_size / GPU_THREAD_COUNT) as u64)
            .arg(&source_buffer)
            .arg(&output_buffer)
            .build()
            .unwrap();

        let expected = s32(generate_data(data_size));
        let mut result = vec![0u32; GPU_THREAD_COUNT as _];

        b.iter(|| {
            unsafe { kernel.enq() }.unwrap();
            output_buffer.read(&mut result).enq().unwrap();
            let result_sum: u32 = result.iter().copied().sum();
            assert_eq!(result_sum, expected);
        });
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("gpu_sum");
    // g.measurement_time(std::time::Duration::from_secs(30));

    let sizes = [1u32]
        .into_iter()
        .chain((0..=(50_000 * 4)).step_by(256 * 32 * 32))
        .chain(((50_000 * 4)..=(50_000 * 64)).step_by(256 * 256 * 10))
        .chain(((0)..=(50_000 * 64 * 64)).step_by(256 * 256 * 12 * 64))
        .filter(|v| *v != 0)
        .map(|v| v.div_ceil(GPU_THREAD_COUNT) * GPU_THREAD_COUNT)
        .unique()
        .sorted()
        .collect_vec();

    println!("{:?}", sizes);

    for data_size in sizes {
        do_cl_bench(
            &mut g,
            data_size,
            include_str!("../shaders/opencl/sum_column_major.cl"),
            "opencl_column_major",
        );
        do_cl_bench(
            &mut g,
            data_size,
            include_str!("../shaders/opencl/sum_row_major.cl"),
            "opencl_row_major",
        );
    }

    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
