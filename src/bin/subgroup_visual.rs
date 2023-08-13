use bytemuck::cast_slice;
use gpu_compute::{
    execute_util::QuadMethod,
    vulkan_util::{MVertex, RenderPassKey, VulkanData},
};
use image::RgbImage;
use itertools::Itertools;
use nalgebra::Vector2;
use sha3::{Digest, Sha3_256};
use std::{
    collections::{BTreeMap, HashMap},
    iter::once,
};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents},
    format::{ClearValue, Format},
    image::view::ImageView,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            rasterization::{PolygonMode, RasterizationState},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::GpuFuture,
};


fn run(vulkan: &mut VulkanData, size: Vector2<u32>, method: QuadMethod) {
    const FORMAT: Format = Format::R16G16_UINT;

    let render_pass = vulkan.create_render_pass(RenderPassKey {
        format: Some(FORMAT),
    });

    let vs = vs::load(vulkan.device.clone()).unwrap();
    let fs = fs::load(vulkan.device.clone()).unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(MVertex::per_vertex())
        .vertex_shader(
            vs.entry_point("main").unwrap(),
            vs::SpecializationConstants {
                DATA_SCALE: if method == QuadMethod::large_triangle {
                    2
                } else {
                    1
                },
            },
        )
        .rasterization_state(RasterizationState::new().polygon_mode(
            if method == QuadMethod::fill_rectangle {
                PolygonMode::FillRectangle
            } else {
                PolygonMode::Fill
            },
        ))
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()))
        .render_pass(subpass)
        .build(vulkan.device.clone())
        .unwrap();

    let mut command_buffer = vulkan.create_command_buffer();

    let target = vulkan.create_target_image(size, FORMAT);
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![ImageView::new_default(target.clone()).unwrap()],
            ..Default::default()
        },
    )
    .unwrap();

    command_buffer
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(ClearValue::Uint([0; 4]))],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassContents::Inline,
        )
        .unwrap()

        // Actual rendering
        .set_viewport(0, once(Viewport {
            origin: [0.0, 0.0],
            dimensions: size.map(|e| e as f32).into(),
            depth_range: 0.0..1.0
        }))
        .bind_pipeline_graphics(pipeline.clone())
        .bind_vertex_buffers(0, vulkan.vertex_buffer())
        .draw(if method == QuadMethod::two_triangles {vulkan.vertex_buffer().len() as _} else {3}, 1, 0, 0)
        .unwrap()

        // End rendering
        .end_render_pass()
        .unwrap();

    let read_buffer: Subbuffer<[[u16; 2]]> =
        vulkan.download_image(&mut command_buffer, target.clone());

    let future = command_buffer
        .build()
        .unwrap()
        .execute(vulkan.queue.clone())
        .unwrap();
    let fence = future.then_signal_fence_and_flush().unwrap();
    fence.wait(None).unwrap();

    let raw_samples = read_buffer.read().unwrap().iter().copied().collect_vec();
    let mut groups = HashMap::new();
    for s in raw_samples.iter().copied() {
        *groups.entry(s).or_insert(0usize) += 1;
    }
    let mut group_counts = BTreeMap::new();
    for s in groups.values().copied() {
        *group_counts.entry(s).or_insert(0usize) += 1;
    }

    // let rg: Vec<u8> = raw_samples.iter().map(|v| [v[0] as u8, v[1] as u8,
    // 0]).flatten().collect_vec(); let rg = RgbImage::from_vec(size.x, size.y,
    // rg).unwrap();

    let hashed: Vec<u8> = raw_samples
        .iter()
        .map(|v| Sha3_256::digest(cast_slice(v)).into_iter().take(3))
        .flatten()
        .collect_vec();
    let hashed = RgbImage::from_vec(size.x, size.y, hashed).unwrap();

    let name = format!("subgroups/subgroups_{}x{}_{:?}", size.x, size.y, method);
    hashed.save(format!("{name}.hashed.png")).unwrap();
    // rg.save(format!("{name}.rg.png")).unwrap();

    std::fs::write(
        format!("{name}.json"),
        serde_json::to_string_pretty(&group_counts).unwrap(),
    )
    .unwrap();
}

fn main() {
    let mut vulkan = VulkanData::init();
    let _ = std::fs::remove_dir_all("subgroups");
    let _ = std::fs::create_dir_all("subgroups");

    for x in [1, 2, 8, 16, 32, 64, 128, 256] {
        for y in [1, 2, 8, 16, 32, 64, 128, 256] {
            for method in QuadMethod::all(&vulkan) {
                run(&mut vulkan, Vector2::new(x, y), *method);
            }
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/basic.vs",
    }
}
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        spirv_version: "1.3",
        path: "shaders/instances/subgroup_find.glsl",
        include: ["shaders/pluggable"],
    }
}
