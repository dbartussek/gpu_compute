pub mod capture;
pub mod execute_util;
pub mod vulkan_util;

use std::ffi::c_int;

#[link(name = "an_external_function")]
extern "C" {
    pub fn an_external_function() -> c_int;
    pub fn do_virtual_call(arg: extern "C" fn() -> c_int) -> c_int;
}
