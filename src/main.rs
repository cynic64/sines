use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use minifb::{Key, Window, WindowOptions};

use image::GenericImageView;

const SCALE: usize = 5;

const PI2: f64 = std::f64::consts::PI * 2.0;

// index with [y * width + x]
#[derive(Debug)]
struct Array2D<T> {
    data: Vec<T>,
    width: usize,
    height: usize,
}

fn main() {
    let input_2d = load_image();

    let st = std::time::Instant::now();
    let output_2d = fourier_2d(&input_2d);
    println!("get fourier: {} ms", get_elapsed(st) * 1_000.0);

    let st = std::time::Instant::now();
    let reconstructed = reconstruct_2d(&output_2d);
    println!("reconstruct: {} ms", get_elapsed(st) * 1_000.0);

    check(&input_2d, &reconstructed);

    display_fb(&input_2d);

    let displayable_output = &powerify(&output_2d);
    display_fb(&displayable_output);

    display_fb(&reconstructed);
}

fn normalize(transform: &mut [Complex<f64>]) {
    let div = transform.len() as f64;

    transform.iter_mut().for_each(|x| {
        x.re /= div;
        x.im /= div;
    });
}

fn reconstruct_2d(input: &Array2D<Complex<f64>>) -> Array2D<f64> {
    let mut output_data = vec![0.0; input.width * input.height];

    input.data.iter().enumerate().for_each(|(idx, point)| {
        let x = idx % input.width;
        let y = idx / input.width;

        let amplitude = (point.re * point.re + point.im * point.im).sqrt();
        let phase = point.im.atan2(point.re);

        // idk why it has to be flipped here (y, x instead of x, y)
        let u = un_negate(y, input.width) as f64;
        let v = un_negate(x, input.height) as f64;

        output_data.iter_mut().enumerate().for_each(|(idx, val)| {
            let x = (idx % input.width) as f64 / input.width as f64;
            let y = (idx / input.width) as f64 / input.height as f64;

            *val += ((u * x + v * y) * PI2 + phase).cos() * amplitude;
        });
    });

    Array2D {
        data: output_data,
        width: input.width,
        height: input.height,
    }
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

fn fourier_2d(input: &Array2D<f64>) -> Array2D<Complex<f64>> {
    // convert everything to complex numbers
    let complex_input: Vec<Complex<f64>> = input
        .data
        .iter()
        .map(|x| Complex { re: *x, im: 0.0 })
        .collect();

    // so that we can compute the FFT of columns later (we do rows first), this
    // is kinda backwards: the initial FFT results are stored vertically, not
    // horizontally
    let mut fourier_columns = vec![Complex::zero(); complex_input.len()];

    complex_input
        .chunks(input.width)
        .enumerate()
        .for_each(|(y, row)| {
            let row_output = fourier(row);

            row_output
                .iter()
                .enumerate()
                // note [x][y] instead of [y][x] - again to be able to compute
                // FFT of columns easily later
                .for_each(|(x, &point)| fourier_columns[x * input.width + y] = point);
        });

    let mut output_data = vec![Complex::zero(); complex_input.len()];

    fourier_columns
        .chunks(input.width)
        .enumerate()
        .for_each(|(y, column)| {
            let column_output = fourier(column);

            column_output
                .iter()
                .enumerate()
                .for_each(|(x, &point)| output_data[y * input.width + x] = point);
        });

    Array2D {
        data: output_data,
        width: input.width,
        height: input.height,
    }
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

fn display_fb(data: &Array2D<f64>) {
    let (window_width, window_height) = (data.width * SCALE, data.height * SCALE);
    let mut buffer: Vec<u32> = vec![0; window_width * window_height];

    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.iter_mut().enumerate().for_each(|(idx, val)| {
            let x = idx % window_width / SCALE;
            let y = idx / window_width / SCALE;
            *val = f64_to_hex(data.data[y * data.width + x])
        });

        window
            .update_with_buffer(&buffer, window_width, window_height)
            .unwrap();
    }
}

fn powerify(input: &Array2D<Complex<f64>>) -> Array2D<f64> {
    let data = input
        .data
        .iter()
        .map(|point| point.re * point.re + point.im * point.im)
        .collect();

    Array2D {
        data,
        width: input.width,
        height: input.height,
    }
}

fn un_negate(x: usize, size: usize) -> f64 {
    if x < size / 2 {
        x as f64
    } else {
        -((size - x) as f64)
    }
}

fn check(a: &Array2D<f64>, b: &Array2D<f64>) {
    let threshold = 0.00001;
    a.data.iter().enumerate().for_each(|(idx, x)| {
        if (x - b.data[idx]).abs() > threshold {
            panic!("Check failed, {} and {} too far apart", x, b.data[idx])
        }
    });
}

fn get_elapsed(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs() as f64 + start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0
}

fn load_image() -> Array2D<f64> {
    let img = image::open("cipi-100x100.png").expect("Couldn't open image");

    let (width, height) = img.dimensions();
    println!("dimensions {}x{}", width, height);

    Array2D {
        data: (0..width * height)
            .map(|idx| {
                let pixel = img.get_pixel(idx % width, idx / width);
                (pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) as f64 / 3.0 / 128.0 - 1.0
            })
            .collect(),
        width: width as usize,
        height: height as usize,
    }
}
