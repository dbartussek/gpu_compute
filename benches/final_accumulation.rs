#![feature(portable_simd)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::Rng;
use rand_pcg::Pcg64;
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

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("final_accumulation");

    let sizes = (4..18u32).into_iter().map(|v| 1u32 << v).collect_vec();
    println!("{:?}", sizes);

    for size in sizes.clone() {
        const DATA_SIZES: usize = 10000000;
        let data = black_box(
            std::iter::repeat_with(|| (1u32..size).into_iter().collect_vec())
                .take(DATA_SIZES / (size as usize))
                .collect_vec(),
        );

        g.bench_with_input(BenchmarkId::new("pcg_cost", size), &size, |b, _size| {
            let mut rng = Pcg64::new(42, 0);

            b.iter(|| {
                let data = data.get(rng.gen_range(0..data.len())).unwrap();
                black_box(data)
            });
        });

        g.bench_with_input(BenchmarkId::new("u32", size), &size, |b, _size| {
            let mut rng = Pcg64::new(42, 0);

            b.iter(|| {
                let data = data.get(rng.gen_range(0..data.len())).unwrap();
                accumulate(black_box(data))
            });
        });

        g.bench_with_input(BenchmarkId::new("u32_vector", size), &size, |b, _size| {
            let mut rng = Pcg64::new(42, 0);

            b.iter(|| {
                let data = data.get(rng.gen_range(0..data.len())).unwrap();
                let mut accumulator = u32x16::from_array([0; 16]);

                for it in black_box(data.chunks_exact(16)) {
                    let it = u32x16::from_array(it.try_into().unwrap());
                    accumulator += it;
                }

                accumulator.reduce_sum()
            });
        });
    }

    drop(g);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
