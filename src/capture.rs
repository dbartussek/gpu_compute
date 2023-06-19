use lazy_static::lazy_static;
use renderdoc::{RenderDoc, V110};
use std::{ffi::c_void, ptr::null, sync::Mutex};

pub fn capture<F, R>(function: F) -> R
where
    F: FnOnce() -> R,
{
    lazy_static! {
        static ref RENDERDOC: Option<Mutex<RenderDoc<V110>>> =
            RenderDoc::<V110>::new().ok().map(Mutex::new);
    }

    let mut renderdoc = (*RENDERDOC).as_ref().map(|r| r.lock().unwrap());

    if let Some(doc) = renderdoc.as_mut() {
        doc.start_frame_capture(null::<c_void>(), null::<c_void>());
    }

    let result = function();

    if let Some(doc) = renderdoc.as_mut() {
        doc.end_frame_capture(null::<c_void>(), null::<c_void>());
    }

    result
}
