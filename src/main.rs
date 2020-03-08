use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use rand::Rng;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 100;
const HEIGHT: usize = 100;
const SCALE: usize = 10;

const LENGTH: usize = 1_000;

fn main() {
    // let mut rng = rand::thread_rng();

    let mut input_1d = [0.0; WIDTH];
    (0..WIDTH)
        .for_each(|x| input_1d[x] = (x as f64 / WIDTH as f64 * 10.0 * std::f64::consts::PI).cos() + (x as f64 / WIDTH as f64 * 30.0 * std::f64::consts::PI).cos() * 0.5);

    let output_1d = fourier(input_1d.to_vec());

    let mut input_2d = [[0.0; WIDTH]; HEIGHT];
    (0..HEIGHT).for_each(|y| (0..WIDTH).for_each(|x| input_2d[y][x] = input_1d[x]));

    let mut output_2d = [[0.0; WIDTH]; HEIGHT];
    (0..HEIGHT).for_each(|y| {
        (0..WIDTH).for_each(|x| {
            let r = output_1d[x].re;
            let i = output_1d[x].im;
            output_2d[y][x] = r * r + i * i;
        })
    });

    display_fb(&input_2d);
    display_fb(&output_2d);
}

#[allow(dead_code)]
fn reconstruct(transform: Vec<Complex<f64>>) -> Vec<(f64, f64)> {
    let threshold = 0.0;
    let half_len = transform.len() / 2;

    // (frequency, fourier point)
    let significant_freqs: Vec<(usize, Complex<f64>)> = (0..half_len)
        .filter_map(|idx| {
            let x = transform[idx];

            if (x.re * x.re + x.im * x.im).sqrt() > threshold {
                Some((idx, x))
            } else {
                None
            }
        })
        .collect();

    let base_level =
        (transform[0].re * transform[0].re + transform[1].im * transform[1].im).sqrt() * 0.5;

    let mut reconstruction = vec![-base_level; LENGTH];

    significant_freqs.iter().for_each(|(freq_idx, x)| {
        // x is the complex result of the fourier transform
        let amplitude = (x.re * x.re + x.im * x.im).sqrt();
        let phase = x.im.atan2(x.re);
        let period = (half_len as f64) / (*freq_idx as f64);

        reconstruction.iter_mut().enumerate().for_each(|(x, y)| {
            *y += (x as f64 / period * std::f64::consts::PI + phase).cos() * amplitude
        });
    });

    reconstruction
        .iter()
        .enumerate()
        .map(|(x, y)| (x as f64, *y))
        .collect()
}

fn normalize(transform: &mut [Complex<f64>]) {
    let div = (transform.len() / 2) as f64;

    transform.iter_mut().for_each(|x| {
        x.re /= div;
        x.im /= div;
    });
}

fn fourier(input: Vec<f64>) -> Vec<Complex<f64>> {
    // calls fft.process and normalizes the output
    let mut processed_input: Vec<_> = input.iter().map(|x| Complex { re: *x, im: 0.0 }).collect();

    let length = input.len();
    let mut output: Vec<Complex<f64>> = vec![Complex::zero(); length];

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(input.len());
    fft.process(&mut processed_input, &mut output);

    normalize(&mut output);

    output
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
    let x = 1.0 / (1.0 + std::f64::consts::E.powf(-x));
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
