#![feature(int_roundings)]

pub mod capture;
pub mod execute_util;
pub mod execute_util_compute;
pub mod vulkan_util;

use itertools::Itertools;
use lazy_static::lazy_static;
use std::ffi::c_int;

#[link(name = "an_external_function")]
extern "C" {
    pub fn an_external_function() -> c_int;
    pub fn do_virtual_call(arg: extern "C" fn() -> c_int) -> c_int;
}

pub const GPU_THREAD_COUNT: u32 = 32 * 256;

lazy_static! {
    pub static ref PROFILING_SIZES: Vec<u32> = [1u32]
        .into_iter()
        // .chain((0..=(50_000 * 4)).step_by(256 * 32 * 32))
        // .chain(((50_000 * 4)..=(50_000 * 64)).step_by(256 * 256 * 10))
        // .chain(((0)..=(50_000 * 64 * 64)).step_by(256 * 256 * 12 * 64))
        .chain((0..29).step_by(2).map(|l| 1 << l))
        .chain([1 << 29])
        .filter(|v| *v != 0)
        .map(|v| v.div_ceil(GPU_THREAD_COUNT) * GPU_THREAD_COUNT)
        .unique()
        .sorted()
        .collect_vec();
}
