use gpu_compute::vulkan_util::VulkanData;
use vulkano::{
    format::Format,
    image::{ImageFormatInfo, ImageTiling, ImageUsage},
};

fn main() {
    let vulkan = VulkanData::init();

    let formats = [
        // i32
        vec![
            Format::R32_SINT,
            Format::R32G32_SINT,
            Format::R32G32B32_SINT,
            Format::R32G32B32A32_SINT,
        ],
        // f32
        vec![
            Format::R32_SFLOAT,
            Format::R32G32_SFLOAT,
            Format::R32G32B32_SFLOAT,
            Format::R32G32B32A32_SFLOAT,
        ],
        // i64
        vec![
            Format::R64_SINT,
            Format::R64G64_SINT,
            Format::R64G64B64_SINT,
            Format::R64G64B64A64_SINT,
        ],
        // f64
        vec![
            Format::R64_SFLOAT,
            Format::R64G64_SFLOAT,
            Format::R64G64B64_SFLOAT,
            Format::R64G64B64A64_SFLOAT,
        ],
    ];
    let tilings = [ImageTiling::Linear, ImageTiling::Optimal];
    let usages = [
        ImageUsage::SAMPLED,
        ImageUsage::STORAGE,
        ImageUsage::COLOR_ATTACHMENT,
    ];

    for group in formats {
        for format in group {
            for usage in usages {
                for tiling in tilings {
                    let id = format!(
                        "{: <20} - {: <16} - {: <7}",
                        format!("{:?}", format),
                        format!("{:?}", usage),
                        format!("{:?}", tiling),
                    );

                    match vulkan
                        .physical_device
                        .image_format_properties(ImageFormatInfo {
                            format: Some(format),
                            usage,
                            tiling,
                            ..Default::default()
                        })
                        .unwrap()
                    {
                        Some(_) => println!("{id} - X"),
                        None => println!("{id} - _"),
                    }
                }
            }
            println!();
        }
        println!();
    }
}
