use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ultrafastgoertzel::{goertzel, goertzel_batch};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate a test signal with a pure sine wave
fn generate_sine_wave(samples: usize, frequency: f64, amplitude: f64) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let x = i as f64;
            amplitude * (2.0 * std::f64::consts::PI * frequency * x).sin()
        })
        .collect()
}

/// Generate a random signal for benchmarking
fn generate_random_signal(samples: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..samples).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// Benchmark the basic goertzel function with different signal sizes including up to 1M samples
fn bench_goertzel_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("goertzel_signal_sizes");
    
    // Test a wide range including very large signals
    for size in [1024, 8192, 65536, 262144, 1_000_000].iter() {
        let signal = generate_sine_wave(*size, 1.0 / 128.0, 1.0);
        let frequency = 1.0 / 128.0;
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| goertzel(black_box(&signal), black_box(frequency)));
        });
    }
    
    group.finish();
}

/// Benchmark goertzel with random noise vs pure sine wave (reduced for speed)
fn bench_goertzel_signal_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("goertzel_signal_types");
    let size = 8192;
    let frequency = 1.0 / 128.0;
    
    let sine_signal = generate_sine_wave(size, frequency, 1.0);
    group.bench_function("sine_wave_8k", |b| {
        b.iter(|| goertzel(black_box(&sine_signal), black_box(frequency)));
    });
    
    let random_signal = generate_random_signal(size, 42);
    group.bench_function("random_noise_8k", |b| {
        b.iter(|| goertzel(black_box(&random_signal), black_box(frequency)));
    });
    
    group.finish();
}

/// Benchmark batch processing with different signal sizes including up to 1M samples
fn bench_goertzel_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("goertzel_batch");
    let frequencies: Vec<f64> = (1..=10).map(|i| i as f64 / 256.0).collect();
    
    // Test batch processing with increasingly large signals
    for size in [1024, 65536, 262144, 1_000_000].iter() {
        let signal = generate_sine_wave(*size, 1.0 / 128.0, 1.0);
        
        group.bench_with_input(
            BenchmarkId::new("batch_10freq_size", size),
            size,
            |b, _| {
                b.iter(|| goertzel_batch(black_box(&signal), black_box(&frequencies)));
            },
        );
    }
    
    group.finish();
}

/// Benchmark single vs batch processing (simplified)
fn bench_single_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_vs_batch");
    let signal = generate_sine_wave(8192, 1.0 / 128.0, 1.0);
    let frequencies: Vec<f64> = (1..=5).map(|i| i as f64 / 256.0).collect();
    
    group.bench_function("single_5_calls", |b| {
        b.iter(|| {
            for freq in &frequencies {
                goertzel(black_box(&signal), black_box(*freq));
            }
        });
    });
    
    group.bench_function("batch_5_frequencies", |b| {
        b.iter(|| goertzel_batch(black_box(&signal), black_box(&frequencies)));
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_goertzel_sizes,
    bench_goertzel_signal_types,
    bench_goertzel_batch,
    bench_single_vs_batch
);
criterion_main!(benches);
