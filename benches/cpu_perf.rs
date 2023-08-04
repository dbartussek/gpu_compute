use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rayon::prelude::*;
use std::{
    hint::black_box,
    ops::{Add, AddAssign},
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
    let mut g = c.benchmark_group("cpu_sum");

    let sizes = [
        1 << 12,
        1 << 14,
        1 << 20,
        1 << 21,
        1 << 21 | 1 << 20,
        1 << 22,
        50_000 * 64 * 10,
        50_000 * 64 * 20,
        50_000 * 64 * 30,
    ];

    for size in sizes {
        g.bench_with_input(BenchmarkId::new("u32", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().collect_vec();

            b.iter(|| accumulate(black_box(&data)));
        });
    }
    for size in sizes {
        g.bench_with_input(BenchmarkId::new("u32_parallel", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().collect_vec();

            b.iter(|| accumulate_parallel(black_box(&data)));
        });
    }

    for size in sizes {
        g.bench_with_input(BenchmarkId::new("f32", size), &size, |b, size| {
            let data = (1u32..*size).into_iter().map(|it| it as f32).collect_vec();

            b.iter(|| accumulate(black_box(&data)));
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
