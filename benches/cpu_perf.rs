#![feature(portable_simd)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gpu_compute::vulkan_util::VulkanData;
use itertools::Itertools;
use rayon::prelude::*;
use std::{
    hint::black_box,
    ops::{Add, AddAssign},
    simd::{u32x16, SimdUint},
};

fn accumulate<T>(i: &[T]) -> T
where
    T: Copy + Default + Add + AddAssign,
{
    let mut acc = Default::default();

    for i in i {
        acc += *i;
    }

    acc
}
fn accumulate_parallel<T>(i: &[T]) -> T
where
    T: Copy + Default + Add<Output = T> + Send + Sync,
{
    i.par_iter().copied().reduce(Default::default, |a, b| a + b)
}

fn criterion_benchmark(c: &mut Criterion) {
    let vulkan = VulkanData::init();

    let mut g = c.benchmark_group("cpu_sum");

    let sizes = vulkan.profiling_sizes()
        .iter()
        .copied()
        // .filter(|size| *size <= (1 << 22))
        .collect_vec();
    println!("{:?}", sizes);

    for size in sizes.clone() {
        g.bench_with_input(BenchmarkId::new("u32", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().collect_vec();

            b.iter(|| accumulate(black_box(&data)));
        });
        g.bench_with_input(BenchmarkId::new("u32_parallel", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().collect_vec();

            b.iter(|| accumulate_parallel(black_box(&data)));
        });

        g.bench_with_input(BenchmarkId::new("u32_vector", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().collect_vec();

            b.iter(|| {
                let mut accumulator = u32x16::from_array([0; 16]);

                for it in black_box(data.chunks_exact(16)) {
                    let it = u32x16::from_array(it.try_into().unwrap());
                    accumulator += it;
                }

                accumulator.reduce_sum()
            });
        });

        g.bench_with_input(
            BenchmarkId::new("u32_parallel_vector", size),
            &size,
            |b, size| {
                let data = (1u32..*size).into_iter().collect_vec();
                let data: Vec<u32x16> = data
                    .chunks_exact(16)
                    .map(|chunk| u32x16::from_array(chunk.try_into().unwrap()))
                    .collect_vec();

                b.iter(|| {
                    black_box(&data)
                        .par_iter()
                        .copied()
                        .reduce(|| u32x16::from_array([0; 16]), |a, b| a + b)
                        .reduce_sum()
                });
            },
        );
    }

    drop(g);

    let mut g = c.benchmark_group("cpu_min");

    for size in sizes.clone() {
        g.bench_with_input(BenchmarkId::new("f32", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().map(|it| it as f32).collect_vec();

            b.iter(|| {
                black_box(&data)
                    .iter()
                    .copied()
                    .reduce(|a, b| if a < b { a } else { b })
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
