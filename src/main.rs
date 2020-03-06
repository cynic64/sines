use plotters::prelude::*;

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

fn main() {
    let mut input: Vec<Complex<f64>> = (0..1_000)
        .map(|x| x as f64 / 100.0)
        .map(|x| (x * std::f64::consts::PI).sin())
        .map(|x| Complex {
            im: 0.0,
            re: x
        })
        .collect();

    let mut output: Vec<Complex<f64>> = vec![Complex::zero(); 1_000];

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(1_000);
    fft.process(&mut input, &mut output);

    let sequence: Vec<_> = output
        .iter()
        .enumerate()
        .map(|(i, x)| (i as f64, x.re))
        .collect();

    plot(sequence);
}

fn plot(sequence: Vec<(f64, f64)>) {
    let root = BitMapBackend::new("out.png", (640, 480)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_ranged(0.0..10.0, -1.0..1.0).unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            sequence,
            &RED,
        )).unwrap()
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();
}
