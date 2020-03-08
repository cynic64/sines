use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 100;
const HEIGHT: usize = 100;
const SCALE: usize = 10;

const PI2: f64 = std::f64::consts::PI * 2.0;

fn main() {
    let mut input_2d = [[0.0; WIDTH]; HEIGHT];
    let u = 1.0;
    let v = 2.0;
    let freq = 10.0;
    (0..HEIGHT).for_each(|wx| {
        (0..WIDTH).for_each(|wy| {
            let x = wx as f64 / WIDTH as f64;
            let y = wy as f64 / WIDTH as f64;

            let n = ((u * (x * freq) + v * (y * freq)) * PI2).cos()
                + ((u * (x * freq) + v * (y * freq)) * PI2).sin();
            input_2d[wy][wx] = n;
        })
    });

    /*
    let mut output_2d = [[0.0; WIDTH]; HEIGHT];
    (0..HEIGHT).for_each(|y| {
        (0..WIDTH).for_each(|x| {
            let r = output_1d[x].re;
            let i = output_1d[x].im;
            output_2d[y][x] = r * r + i * i;
        })
    });
    */

    display_fb(&input_2d);
    // display_fb(&output_2d);
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
