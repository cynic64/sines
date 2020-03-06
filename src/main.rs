use plotters::prelude::*;

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use rand::Rng;

const LENGTH: usize = 1000;

fn main() {
    let mut _input: Vec<Complex<f64>> = (0..LENGTH)
        .map(|x| x as f64 / 10.0)
        .map(|x| (x * std::f64::consts::PI + 0.1).cos() + 0.2 * (x * 0.5 * std::f64::consts::PI).cos())
        .map(|x| Complex {
            im: 0.0,
            re: x
        })
        .collect();

    let mut rng = rand::thread_rng();

    let mut input: Vec<Complex<f64>> = (0..LENGTH)
        .map(|_| Complex {
            im: 0.0,
            re: rng.gen(),
        })
        .collect();

    let input_sequence: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, x)| (i as f64, x.re))
        .collect();

    let mut output: Vec<Complex<f64>> = vec![Complex::zero(); LENGTH];

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(LENGTH);
    fft.process(&mut input, &mut output);

    normalize(&mut output);

    let fourier_sequence: Vec<_> = output
        .iter()
        .enumerate()
        .map(|(i, x)| (i as f64, (x.re * x.re + x.im * x.im).sqrt() / LENGTH as f64))
        .collect();

    let reconstruction = reconstruct(output);

    plot(input_sequence, fourier_sequence, reconstruction);
}

fn plot(sequence1: Vec<(f64, f64)>, sequence2: Vec<(f64, f64)>, sequence3: Vec<(f64, f64)>) {
    let root = BitMapBackend::new("out.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_ranged(0.0..100.0, -2.0..2.0).unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            sequence1,
            &RED,
        )).unwrap();

    chart
        .draw_series(LineSeries::new(
            sequence2,
            &GREEN,
        )).unwrap();

    chart
        .draw_series(LineSeries::new(
            sequence3,
            &BLUE,
        )).unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();
}

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

    let base_level = (transform[0].re * transform[0].re + transform[1].im * transform[1].im).sqrt() * 0.5;

    let mut reconstruction = vec![-base_level; LENGTH];

    significant_freqs
        .iter()
        .for_each(|(freq_idx, x)| {
            // x is the complex result of the fourier transform
            let amplitude = (x.re * x.re + x.im * x.im).sqrt();
            let phase = x.im.atan2(x.re);
            let period = (half_len as f64) / (*freq_idx as f64);

            reconstruction
                .iter_mut()
                .enumerate()
                .for_each(|(x, y)| *y += (x as f64 / period * std::f64::consts::PI + phase).cos() * amplitude);
        });

    reconstruction
        .iter()
        .enumerate()
        .map(|(x, y)| (x as f64, *y))
        .collect()
}

fn normalize(transform: &mut [Complex<f64>]) {
    let div = (transform.len() / 2) as f64;

    transform
        .iter_mut()
        .for_each(|x| {
            x.re /= div;
            x.im /= div;
        });
}
