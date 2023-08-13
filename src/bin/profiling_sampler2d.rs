#![feature(int_roundings)]

use gpu_compute::{
    execute_util::{ExecuteUtil, OutputKind, QuadMethod},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;
use std::{sync::Arc, time::Duration};
use vulkano::{
    image::ImageUsage,
    swapchain::{acquire_next_image, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::GpuFuture,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

fn main() {
    let mut vulkan = VulkanData::init();

    let event_loop = EventLoop::new();
    let window = Arc::new(Window::new(&event_loop).unwrap());

    let surface =
        vulkano_win::create_surface_from_winit(window.clone(), vulkan.instance.clone()).unwrap();
    let (sc, _images) = Swapchain::new(
        vulkan.device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: 2,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
    )
    .unwrap();

    let data_size = Vector2::<u32>::new(256, 256);
    let work_size_y = 16u32;

    let shader = attach_none_sampled_loop::load(vulkan.device.clone()).unwrap();
    let mut execute = ExecuteUtil::<u32>::setup_2d_sampler(
        &mut vulkan,
        data_size,
        &shader,
        attach_none_sampled_loop::SpecializationConstants {
            TEXTURE_SIZE_X: (data_size.x / work_size_y) as i32,
            TEXTURE_SIZE_Y: 1,
        },
        OutputKind::Attachment,
        QuadMethod::large_triangle,
        work_size_y,
        |a, b| a + b,
    );

    event_loop.run(move |e, _, control| match e {
        Event::NewEvents(_) => {},
        Event::WindowEvent {
            window_id: _,
            event,
        } => {
            if event == WindowEvent::CloseRequested {
                control.set_exit();
            }
        },
        Event::DeviceEvent { .. } => {},
        Event::UserEvent(_) => {},
        Event::Suspended => {},
        Event::Resumed => {},
        Event::MainEventsCleared => {
            execute.run(&mut vulkan, true);

            let (index, _, future) = acquire_next_image(sc.clone(), None).unwrap();
            let future = future
                .then_swapchain_present(
                    vulkan.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(sc.clone(), index),
                )
                .then_signal_fence_and_flush()
                .unwrap();
            future.wait(None).unwrap();

            std::thread::sleep(Duration::from_millis(10));
            control.set_exit();
        },
        Event::RedrawRequested(_) => {},
        Event::RedrawEventsCleared => {},
        Event::LoopDestroyed => {},
    });
}

mod attach_none_sampled_loop {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/instances/gpu_sum/attach_none_sampled2D_loop.glsl",
        include: ["shaders/pluggable"],
    }
}
