use criterion::{criterion_group, criterion_main};

#[cfg(feature = "cuda")]
mod imp {
    use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion};
    use gpu_compute::{execute_util::generate_data, vulkan_util::VulkanData};
    use itertools::Itertools;
    use std::time::Duration;

    pub enum Algo {
        Simple,
        Subgroup,
    }

    fn do_cuda_bench(
        g: &mut BenchmarkGroup<WallTime>,
        data_size: u32,
        kernel_size: u32,
        name: &str,
        algo: Algo,
    ) {
        g.bench_with_input(BenchmarkId::new(name, data_size), &data_size, |b, _| {
            let data = generate_data::<u32>(data_size).collect_vec();
            unsafe { gpu_compute::cuda_accumulate_u32_set_data(data.as_ptr(), data.len()) };

            match algo {
                Algo::Simple => {
                    b.iter(|| {
                        let result_sum = unsafe {
                            gpu_compute::cuda_accumulate_u32_sum(
                                kernel_size as usize,
                                kernel_size.min(256) as usize,
                                0,
                            )
                        };
                        // assert_eq!(result_sum, expected);
                        result_sum
                    });
                },
                Algo::Subgroup => {
                    b.iter(|| unsafe { gpu_compute::cuda_accumulate_u32_sum_subgroup() })
                },
            }
        });
    }


    pub fn criterion_benchmark(c: &mut Criterion) {
        let vulkan = VulkanData::init();

        let sizes = vulkan.profiling_sizes().clone();
        println!("{:X?}", sizes);

        {
            let mut g = c.benchmark_group("call_times");
            g.measurement_time(Duration::from_secs(30));
            g.sample_size(1000);

            do_cuda_bench(&mut g, 1, 1, "cuda", Algo::Simple);
        }

        {
            let mut g = c.benchmark_group("gpu_sum");
            // g.measurement_time(std::time::Duration::from_secs(30));
            g.sample_size(10);

            for data_size in sizes.clone() {
                do_cuda_bench(&mut g, data_size, 1024, "cuda", Algo::Simple);
                do_cuda_bench(&mut g, data_size, 1024, "cuda_subgroup", Algo::Subgroup);
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod imp {
    use criterion::Criterion;
    pub fn criterion_benchmark(_: &mut Criterion) {}
}

criterion_group!(benches, imp::criterion_benchmark);
criterion_main!(benches);
