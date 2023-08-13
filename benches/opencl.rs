#![feature(int_roundings)]

use bytemuck::Pod;
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use gpu_compute::{execute_util::generate_data, GPU_THREAD_COUNT, PROFILING_SIZES};
use itertools::Itertools;
use num::{NumCast, Zero};
use ocl::{r#async::BufferSink, Buffer, MemFlags, OclPrm, ProQue, WriteGuard};
use ocl_futures::future::Future;
use std::{fmt::Debug, time::Duration};
use vulkano::buffer::BufferContents;

pub fn do_cl_bench<Type, Acc>(
    g: &mut BenchmarkGroup<WallTime>,
    data_size: u32,
    kernel_size: u32,
    src: &str,
    name: &str,

    accumulate: Acc,
) where
    Type: Copy + NumCast + Pod + BufferContents + PartialEq + Debug + OclPrm + Zero,
    Acc: Fn(Type, Type) -> Type,
{
    g.bench_with_input(BenchmarkId::new(name, data_size), &data_size, |b, _| {
        let cl_program = ProQue::builder()
            .src(src)
            .dims(kernel_size)
            .build()
            .unwrap();

        let source_buffer = Buffer::<Type>::builder()
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

            let source_sink: BufferSink<Type> = WriteGuard::release(writer).into();
            source_sink.flush().enq().unwrap().wait().unwrap();
        }

        let output_buffer = cl_program
            .buffer_builder::<Type>()
            .flags(MemFlags::empty().host_read_only().write_only())
            .build()
            .unwrap();

        let kernel = cl_program
            .kernel_builder("sum")
            .arg(kernel_size as u64)
            .arg((data_size / kernel_size) as u64)
            .arg(&source_buffer)
            .arg(&output_buffer)
            .build()
            .unwrap();

        let expected: Type = generate_data(data_size).reduce(&accumulate).unwrap();
        let mut result = vec![Type::zero(); kernel_size as _];

        b.iter(|| {
            unsafe { kernel.enq() }.unwrap();
            output_buffer.read(&mut result).enq().unwrap();
            let result_sum = black_box(result.iter().copied().reduce(&accumulate)).unwrap();
            assert_eq!(result_sum, expected);
        });
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let sizes = PROFILING_SIZES.clone();
    println!("{:X?}", sizes);

    {
        let mut g = c.benchmark_group("call_times");
        g.measurement_time(Duration::from_secs(30));
        g.sample_size(1000);

        do_cl_bench::<u32, _>(
            &mut g,
            1,
            1,
            include_str!("../shaders/opencl/sum_column_major.cl"),
            "opencl",
            |a, b| a + b,
        );
    }

    {
        let mut g = c.benchmark_group("gpu_sum");
        // g.measurement_time(std::time::Duration::from_secs(30));
        g.sample_size(10);

        for data_size in sizes.clone() {
            do_cl_bench::<u32, _>(
                &mut g,
                data_size,
                GPU_THREAD_COUNT,
                include_str!("../shaders/opencl/sum_column_major.cl"),
                "opencl_column_major",
                |a, b| a + b,
            );
            do_cl_bench::<u32, _>(
                &mut g,
                data_size,
                GPU_THREAD_COUNT,
                include_str!("../shaders/opencl/sum_row_major.cl"),
                "opencl_row_major",
                |a, b| a + b,
            );
        }
    }
    {
        let mut g = c.benchmark_group("gpu_min_f32");
        // g.measurement_time(std::time::Duration::from_secs(30));
        g.sample_size(10);

        for data_size in sizes.clone() {
            do_cl_bench::<f32, _>(
                &mut g,
                data_size,
                GPU_THREAD_COUNT,
                include_str!("../shaders/opencl/min_column_major_f32.cl"),
                "opencl_column_major",
                |a, b| a.min(b),
            );
            do_cl_bench::<f32, _>(
                &mut g,
                data_size,
                GPU_THREAD_COUNT,
                include_str!("../shaders/opencl/min_row_major_f32.cl"),
                "opencl_row_major",
                |a, b| a.min(b),
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
