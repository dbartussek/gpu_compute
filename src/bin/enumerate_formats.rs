use gpu_compute::vulkan_util::VulkanData;
use vulkano::{
    format::{Format, FormatFeatures},
    image::{ImageFormatInfo, ImageTiling, ImageUsage},
};

fn main() {
    let vulkan = VulkanData::init();

    let formats = [
        // u8
        vec![
            Format::R8_UINT,
            Format::R8G8_UINT,
            Format::R8G8B8_UINT,
            Format::R8G8B8A8_UINT,
        ],
        // i8
        vec![
            Format::R8_SINT,
            Format::R8G8_SINT,
            Format::R8G8B8_SINT,
            Format::R8G8B8A8_SINT,
        ],
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
                        Some(_) => {
                            let details = match vulkan
                                .physical_device
                                .format_properties(format)
                                .ok()
                                .filter(|_| usage == ImageUsage::COLOR_ATTACHMENT)
                            {
                                None => format!(""),
                                Some(props) => format!(
                                    " {}",
                                    if match tiling {
                                        ImageTiling::Linear => props
                                            .linear_tiling_features
                                            .contains(FormatFeatures::COLOR_ATTACHMENT_BLEND),
                                        ImageTiling::Optimal => props
                                            .optimal_tiling_features
                                            .contains(FormatFeatures::COLOR_ATTACHMENT_BLEND),
                                        _ => unreachable!(),
                                    } {
                                        "can blend"
                                    } else {
                                        ""
                                    }
                                ),
                            };
                            println!("{id} - X{}", details);
                        },
                        None => println!("{id} - _"),
                    }
                }
            }
            println!();
        }
        println!();
    }
}
