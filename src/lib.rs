#![feature(int_roundings)]

pub mod capture;
pub mod execute_util;
pub mod execute_util_compute;
pub mod vulkan_util;

use std::ffi::c_int;

#[link(name = "an_external_function")]
extern "C" {
    pub fn an_external_function() -> c_int;
    pub fn do_virtual_call(arg: extern "C" fn() -> c_int) -> c_int;
}


#[cfg(feature = "cuda")]
#[link(name = "cuda_accumulate", kind = "static")]
extern "C" {
    pub fn cuda_empty_kernel() -> std::ffi::c_void;

    pub fn cuda_accumulate_u32_set_data(data: *const u32, data_size: usize) -> std::ffi::c_void;
    pub fn cuda_accumulate_u32_sum(
        total_threads: usize,
        subgroup_size: usize,
        second_accumulate_on_gpu: usize,
    ) -> u32;
    pub fn cuda_accumulate_u32_sum_subgroup() -> u32;
}
