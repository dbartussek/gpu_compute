use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
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

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("cpu_sum/u32");

    for size in [
        1 << 12,
        1 << 14,
        1 << 20,
        1 << 21,
        1 << 21 | 1 << 20,
        1 << 22,
        4000000,
    ] {
        g.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, size| {
            let data = (1u32..*size).into_iter().collect_vec();

            b.iter(|| accumulate(black_box(&data)));
        });
    }
    drop(g);

    let mut g = c.benchmark_group("cpu_sum/f32");
    for size in [
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 17,
        1 << 18,
        1 << 19,
        1 << 20,
        300000,
        350000,
    ] {
        g.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, size| {
            let data = (1u32..*size).into_iter().map(|it| it as f32).collect_vec();

            b.iter(|| accumulate(black_box(&data)));
        });
    }
    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
