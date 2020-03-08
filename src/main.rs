use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use minifb::{Key, Window, WindowOptions};

use rand::{Rng, thread_rng};

const WIDTH: usize = 100;
const HEIGHT: usize = 100;
const SCALE: usize = 10;

const PI2: f64 = std::f64::consts::PI * 2.0;

fn main() {
    let mut rng = thread_rng();

    let mut input_2d = [[0.0; WIDTH]; HEIGHT];
    let u = 1.0;
    let v = 1.0;
    let freq = 50.0;
    (0..HEIGHT).for_each(|wx| {
        (0..WIDTH).for_each(|wy| {
            /*
            let x = wx as f64 / WIDTH as f64;
            let y = wy as f64 / WIDTH as f64;

            let n = ((u * (x * freq) + v * (y * freq)) * PI2).cos()
                + ((u * (x * freq) + v * (y * freq)) * PI2).sin();

            // let n = rng.gen();
            input_2d[wy][wx] = n;
             */

            let x = wx as f64 / WIDTH as f64;
            let y = wy as f64 / WIDTH as f64;
            let d = ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)).sqrt();

            if d < 0.1 {
                input_2d[wy][wx] = 1.0;
            } else {
                input_2d[wy][wx] = 0.0;
            }
        })
    });

    let output_2d = fourier_2d(&input_2d);

    display_fb(&input_2d);
    display_fb(&output_2d);
}

fn normalize(transform: &mut [Complex<f64>]) {
    let div = (transform.len() / 2) as f64;

    transform.iter_mut().for_each(|x| {
        x.re /= div;
        x.im /= div;
    });
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

    let half_len = output.len() / 2;

    (0..input.len())
        .map(|idx| if idx < half_len {
            output[half_len - idx - 1]
        } else {
            output[idx - half_len]
        })
        .collect()
}

fn fourier_2d(input: &[[f64; WIDTH]; HEIGHT]) -> [[f64; WIDTH]; HEIGHT] {
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

    let mut output_2d = [[0.0; WIDTH]; HEIGHT];

    fourier_columns.iter().enumerate().for_each(|(y, column)| {
        let column_output = fourier(column);

        column_output
            .iter()
            .enumerate()
            .for_each(|(x, &point)| output_2d[y][x] = point.re * point.re + point.im * point.im);
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
    let contrast_factor = 10_000.0;

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
