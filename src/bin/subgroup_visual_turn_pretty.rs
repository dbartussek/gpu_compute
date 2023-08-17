use clap::Parser;
use image::{GenericImageView, Pixel, Rgb, RgbImage};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
}

fn main() {
    const TILE_SIZE: u32 = 4;

    let args = Args::parse();

    let input = image::open(&args.input).unwrap();
    let mut output = RgbImage::new(
        input.width() * TILE_SIZE + (input.width() - 1),
        input.height() * TILE_SIZE + (input.height() - 1),
    );
    output.pixels_mut().for_each(|px| *px = Rgb([u8::MAX; 3]));

    for y in 0..input.height() {
        for x in 0..input.width() {
            for dx in 0..TILE_SIZE {
                for dy in 0..TILE_SIZE {
                    output.put_pixel(
                        (x * (TILE_SIZE + 1)) + dx,
                        (y * (TILE_SIZE + 1)) + dy,
                        input.get_pixel(x, y).to_rgb(),
                    );
                }
            }
        }
    }


    for y in (2..input.height()).step_by(2) {
        for x in 0..output.width() {
            output.put_pixel(
                x,
                (y * (TILE_SIZE + 1)) - 1,
                Rgb([0; 3]),
            );
        }
    }
    for x in (2..input.width()).step_by(2) {
        for y in 0..output.height() {
            output.put_pixel(
                (x * (TILE_SIZE + 1)) - 1,
                y,
                Rgb([0; 3]),
            );
        }
    }

    output.save(&args.output).unwrap();
}
