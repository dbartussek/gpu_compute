#![feature(int_roundings)]

use clap::Parser;
use gpu_compute::{
    execute_util::{ExecuteParameters, ExecuteUtil, OutputKind, QuadMethod},
    vulkan_util::VulkanData,
};
use nalgebra::Vector2;
use std::{io::stdin, sync::Arc, time::Duration};
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
    pub separate_read_buffer: bool,
    #[arg(short, long, default_value_t = false)]
    pub exit: bool,

    #[arg(short, long, default_value_t = 1)]
    pub framebuffer_y: u32,

    #[arg(short, long, default_value_t = 100_000_000u32)]
    pub data_size: u32,
}

fn main() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        hook(p);
        std::fs::write("exit.txt", format!("Panic {:#?}", p)).unwrap();
        let _ = stdin().read_line(&mut Default::default());
    }));

    let args = Args::parse();

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

    let data_size = Vector2::<u32>::new(16384, args.data_size.div_ceil(16384));
    println!("{:?}", data_size);

    let shader = attach_none_sampled_loop::load(vulkan.device.clone()).unwrap();
    let mut execute = ExecuteUtil::<u32>::setup_2d_sampler(
        &mut vulkan,
        data_size,
        &shader,
        attach_none_sampled_loop::SpecializationConstants {
            TEXTURE_SIZE_X: (data_size.x / args.framebuffer_y) as _,
            TEXTURE_SIZE_Y: args.framebuffer_y as _,
        },
        ExecuteParameters {
            output: OutputKind::Attachment,
            quad_method: QuadMethod::large_triangle,
            framebuffer_y: args.framebuffer_y,
            ..Default::default()
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
                std::fs::write("exit.txt", format!("Window closed")).unwrap();
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

            if args.exit {
                std::fs::write("exit.txt", format!("Immediate exit")).unwrap();
                control.set_exit();
            } else {
                std::thread::sleep(Duration::from_millis(10));
                control.set_exit();
            }
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