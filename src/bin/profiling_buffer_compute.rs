#![feature(int_roundings)]

use clap::Parser;
use gpu_compute::{
    execute_util_compute::{ComputeExecuteUtil, ComputeParameters},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;
use std::{sync::Arc, time::Duration};
use std::io::stdin;
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    separate_read_buffer: bool,

    #[arg(short, long, default_value_t = false)]
    run_once: bool,

    #[arg(short, long)]
    invocation_size: u32,
    #[arg(short, long)]
    work_size: u32,
}

fn main() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        hook(p);
        std::fs::write("exit.txt", format!("Panic {:#?}", p)).unwrap();
        let _ = stdin().read_line(&mut Default::default());
    }));

    let args = Args::try_parse();
    println!("Args: {:#?}", args);
    if args.is_err() {
        let _ = stdin().read_line(&mut Default::default());
    }
    let args = args.unwrap();


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

    let data_size = Vector2::<u32>::new(args.invocation_size, args.work_size.div_ceil(args.invocation_size));

    let shader = compute_none_sbuffer_loop::load(vulkan.device.clone()).unwrap();
    let mut execute = ComputeExecuteUtil::<u32>::setup_storage_buffer(
        &mut vulkan,
        data_size,
        &shader,
        compute_none_sbuffer_loop::SpecializationConstants {
            TEXTURE_SIZE_X: data_size.x as _,
            TEXTURE_SIZE_Y: 1,
        },
        ComputeParameters {
            ..ComputeParameters::default()
        },
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
            execute.run(&mut vulkan, args.separate_read_buffer);

            let (index, _, future) = acquire_next_image(sc.clone(), None).unwrap();
            let future = future
                .then_swapchain_present(
                    vulkan.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(sc.clone(), index),
                )
                .then_signal_fence_and_flush()
                .unwrap();
            future.wait(None).unwrap();

            if args.run_once {
                control.set_exit();
            } else {
                std::thread::sleep(Duration::from_millis(10));
            }
        },
        Event::RedrawRequested(_) => {},
        Event::RedrawEventsCleared => {},
        Event::LoopDestroyed => {},
    });
}

mod compute_none_sbuffer_loop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/instances/gpu_sum/buffer_none_sbuffer_loop.glsl",
        include: ["shaders/pluggable"],
        define: [("COMPUTE_SHADER", "1")],
    }
}
