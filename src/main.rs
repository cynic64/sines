use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use minifb::{Key, Window, WindowOptions};

use rand::{thread_rng, Rng};

use image::{DynamicImage, GenericImage, GenericImageView};

const WIDTH: usize = 200;
const HEIGHT: usize = 200;
const SCALE: usize = 5;

const PI2: f64 = std::f64::consts::PI * 2.0;

fn main() {
    let img = image::open("cipi-200x200.png").expect("Couldn't open image");

    println!("dimensions {:?}", img.dimensions());

    let mut input_2d = [[0.0; WIDTH]; HEIGHT];

    let st = std::time::Instant::now();
    (0..HEIGHT).for_each(|y| {
        (0..WIDTH).for_each(|x| {
            let pixel = img.get_pixel(x as u32, y as u32);
            input_2d[y][x] = (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) as f64 / 3.0 / 128.0 - 1.0;
        })
    });
    println!("set input: {} ms", get_elapsed(st) * 1_000.0);

    let st = std::time::Instant::now();
    let output_2d = fourier_2d(&input_2d);
    println!("get fourier: {} ms", get_elapsed(st) * 1_000.0);

    let st = std::time::Instant::now();
    let reconstructed = reconstruct_2d(&output_2d);
    println!("reconstruct: {} ms", get_elapsed(st) * 1_000.0);

    /*
    check(&input_2d, &reconstructed);

    display_fb(&input_2d);

    let displayable_output = &powerify(&output_2d);
    display_fb(&displayable_output);

    display_fb(&reconstructed);
    */
}

fn normalize(transform: &mut [Complex<f64>]) {
    let div = transform.len() as f64;

    transform.iter_mut().for_each(|x| {
        x.re /= div;
        x.im /= div;
    });
}

fn reconstruct_2d(input: &[[Complex<f64>; WIDTH]; HEIGHT]) -> [[f64; WIDTH]; HEIGHT] {
    let threshold = 0.1;

    let mut output_2d = [[0.0; WIDTH]; HEIGHT];

    (0..HEIGHT).for_each(|y| {
        (0..WIDTH).for_each(|x| {
            let point = input[y][x];
            let amplitude = (point.re * point.re + point.im * point.im).sqrt();
            let phase = point.im.atan2(point.re);

            // idk why
            /*
            let flip = if phase <= 0.0 {
                1.0
            } else {
                -1.0
            };
            */

            // idk why it has to be flipped here
            let u = un_negate(y, WIDTH) as f64;
            let v = un_negate(x, HEIGHT) as f64;

            if amplitude > threshold {
                /*
                println!(
                    "Significant frequency at x: {}, y: {}, amplitude: {}, phase: {}, u: {}, v: {}",
                    x, y, amplitude, phase, u, v,
                );
                */
            }

            output_2d.iter_mut().enumerate().for_each(|(wy, row)| {
                row.iter_mut().enumerate().for_each(|(wx, val)| {
                    let x = wx as f64 / WIDTH as f64;
                    let y = wy as f64 / WIDTH as f64;

                    /*
                     *val += (((u * x + v * y) * PI2).cos() + ((u * x + v * y) * PI2).sin())
                     * amplitude;
                     */
                    *val += ((u * x + v * y) * PI2 + phase).cos() * amplitude;
                })
            });
        })
    });

    output_2d
}

fn fourier(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    // odd-numbered inputs will cause problems
    assert!(input.len() % 2 == 0);

    let mut our_input = input.to_vec();

    // calls fft.process and normalizes the output
    let length = input.len();
    let mut output: Vec<Complex<f64>> = vec![Complex::zero(); length];

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(input.len());
    fft.process(&mut our_input, &mut output);

    normalize(&mut output);

    output
}

fn fourier_2d(input: &[[f64; WIDTH]; HEIGHT]) -> [[Complex<f64>; WIDTH]; HEIGHT] {
    // convert everything to complex numbers
    let complex_input: Vec<Vec<Complex<f64>>> = input
        .iter()
        .map(|row| row.iter().map(|&x| Complex { re: x, im: 0.0 }).collect())
        .collect();

    // so that we can compute the FFT of columns later (we do rows first), this
    // is kinda backwards: the initial FFT results are stored vertically, not
    // horizontally
    let mut fourier_columns = [[Complex::zero(); WIDTH]; HEIGHT];

    complex_input.iter().enumerate().for_each(|(y, row)| {
        let row_output = fourier(row);

        row_output
            .iter()
            .enumerate()
            // note [x][y] instead of [y][x] - again to be able to compute
            // FFT of columns easily later
            .for_each(|(x, &point)| fourier_columns[x][y] = point);
    });

    let mut output_2d = [[Complex::zero(); WIDTH]; HEIGHT];

    fourier_columns.iter().enumerate().for_each(|(y, column)| {
        let column_output = fourier(column);

        column_output
            .iter()
            .enumerate()
            .for_each(|(x, &point)| output_2d[y][x] = point);
    });

    output_2d
}

fn hex(r: u32, g: u32, b: u32) -> u32 {
    assert!(r < 256);
    assert!(g < 256);
    assert!(b < 256);
    r * 65536 + g * 256 + b
}

fn ghex(v: u32) -> u32 {
    hex(v, v, v)
}

fn f64_to_hex(x: f64) -> u32 {
    // let contrast_factor = 10_000.0;
    let contrast_factor = 10.0;

    let x = 1.0 / (1.0 + std::f64::consts::E.powf(-(x * contrast_factor)));

    let mut v = ((x + 1.0) * 127.0) as i32;

    if v > 255 {
        v = 255
    } else if v < 0 {
        v = 0
    }

    ghex(v as u32)
}

fn display_fb(data: &[[f64; WIDTH]; HEIGHT]) {
    let mut buffer: Vec<u32> = vec![0; WIDTH * SCALE * HEIGHT * SCALE];

    let mut window = Window::new(
        "Test - ESC to exit",
        WIDTH * SCALE,
        HEIGHT * SCALE,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        for (idx, val) in buffer.iter_mut().enumerate() {
            let x = idx % (WIDTH * SCALE) / SCALE;
            let y = idx / (WIDTH * SCALE) / SCALE;

            *val = f64_to_hex(data[y][x]);
        }

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&buffer, WIDTH * SCALE, HEIGHT * SCALE)
            .unwrap();
    }
}

fn powerify(data: &[[Complex<f64>; WIDTH]; HEIGHT]) -> [[f64; WIDTH]; HEIGHT] {
    let mut output = [[0.0; WIDTH]; HEIGHT];

    data.iter().enumerate().for_each(|(y, row)| {
        row.iter()
            .enumerate()
            .for_each(|(x, point)| output[y][x] = point.re * point.re + point.im * point.im)
    });

    output
}

fn flip(x: usize, size: usize) -> usize {
    if x < size / 2 {
        size / 2 - x - 1
    } else {
        size - x + size / 2 - 1
    }
}

fn un_negate(x: usize, size: usize) -> f64 {
    if x < size / 2 {
        x as f64
    } else {
        -((size - x) as f64)
    }
}

fn check(a: &[[f64; WIDTH]; HEIGHT], b: &[[f64; WIDTH]; HEIGHT]) {
    let threshold = 0.0001;

    a.iter().enumerate().for_each(|(y, row)| {
        row.iter().enumerate().for_each(|(x, val)| {
            if (val - b[y][x]).abs() > threshold {
                println!("diff: {}", (val - b[y][x]).abs());
            }
        })
    })
}

// flips DC to center and nyquist to border
fn fold(input: &[[f64; WIDTH]; HEIGHT]) -> [[f64; WIDTH]; HEIGHT] {
    let mut output = [[0.0; WIDTH]; HEIGHT];

    input.iter().enumerate().for_each(|(y, row)| {
        row.iter()
            .enumerate()
            .for_each(|(x, val)| output[flip(y, HEIGHT)][flip(x, WIDTH)] = *val)
    });

    output
}

pub fn get_elapsed(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs() as f64 + start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0
}
