use pulp::Simd;

pub fn goertzel(signal: &[f64], frequency: f64) -> f64 {
    return goertzel_batch(signal, &[frequency])[0];
}

#[pulp::with_simd(goertzel_batch = pulp::Arch::new())]
#[inline(always)]
pub fn goertzel_batch_with_simd<'a, S: Simd>(
    simd: S,
    signal: &'a [f64],
    frequencies: &'a [f64],
) -> Vec<f64> {
    frequencies.iter().for_each(|&frequency| {
        assert!(
            frequency >= 0.0 && frequency <= 0.5,
            "frequency must be in range [0.0, 0.5]"
        )
    });
    if signal.len() == 0 {
        return frequencies.iter().map(|_| 0.0).collect();
    }
    // let (head, tail) = S::as_mut_simd_f64s(frequencies);
    let n = signal.len() as f64;
    let w_left_side = 2.0 * std::f64::consts::PI / n;
    let mut coeffs = frequencies
        .into_iter()
        .map(|&freq| {
            let k = freq * n;
            let w = w_left_side * k;
            let cos_w = w.cos();
            2.0 * cos_w
        })
        .collect::<Vec<_>>();
    let (head, tail) = S::as_mut_simd_f64s(&mut coeffs);
    let mut all_magnitudes = vec![0.0; frequencies.len()];

    let mut offset = 0;
    for coeff in head {
        let mut s1 = simd.splat_f64s(0.0);
        let mut s2 = simd.splat_f64s(0.0);
        for &sample in signal {
            let s0 = simd.add_f64s(
                simd.sub_f64s(simd.mul_f64s(*coeff, s1), s2),
                simd.splat_f64s(sample),
            );
            s2 = s1;
            s1 = s0;
        }
        let magnitudes = simd.sub_f64s(
            simd.add_f64s(simd.mul_f64s(s1, s1), simd.mul_f64s(s2, s2)),
            simd.mul_f64s(simd.mul_f64s(s1, s2), *coeff),
        );
        // somehow here add the  magnitudes to all_magnitudes
        simd.partial_store_f64s(&mut all_magnitudes[offset..], magnitudes);
        offset += S::F64_LANES;
    }

    let mut s0: f64;

    for &mut coeff in tail {
        let (mut s1, mut s2) = (0.0, 0.0);
        for &sample in signal {
            s0 = coeff * s1 - s2 + sample;
            s2 = s1;
            s1 = s0;
        }
        let magnitude = s1 * s1 + s2 * s2 - s1 * s2 * coeff;
        all_magnitudes[offset] = magnitude;
        offset += 1;
    }

    let normalizer = n / 2.0;

    all_magnitudes
        .iter()
        .map(|v| v.sqrt() / normalizer)
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave signal
    fn wave(amp: f64, freq: f64, phase: f64, samples: usize) -> Vec<f64> {
        (0..samples)
            .map(|i| {
                let x = i as f64;
                amp * (2.0 * std::f64::consts::PI * freq * x + phase).sin()
            })
            .collect()
    }

    #[test]
    fn test_pure_sine_wave() {
        // Generate test signal
        let n_samples = 512;
        let frequency = 1.0 / 128.0; // Normalized frequency
        let amplitude_expected = 2.5;
        let phase_expected = std::f64::consts::PI / 4.0;

        let signal = wave(amplitude_expected, frequency, phase_expected, n_samples);

        // Run Goertzel algorithm
        let amp = goertzel(&signal, frequency);

        // Check results (with tolerance for floating-point)
        assert!(
            (amp - amplitude_expected).abs() < 1e-10,
            "Expected amplitude {}, got {}",
            amplitude_expected,
            amp
        );
    }

    #[test]
    fn test_multiple_frequencies() {
        let n_samples = 1024;

        // Create composite signal
        let freqs = [
            1.0 / 256.0,
            1.0 / 128.0,
            1.0 / 64.0,
            1.0 / 32.0,
            1.0 / 16.0,
            1.0 / 8.0,
        ];
        let amps = [1.0, 2.0, 0.5, 0.25, 0.125, 0.0625];

        let mut signal = vec![0.0; n_samples];
        for (amp, freq) in amps.iter().zip(freqs.iter()) {
            let component = wave(*amp, *freq, 0.0, n_samples);
            for (i, val) in component.iter().enumerate() {
                signal[i] += val;
            }
        }

        // Test each frequency
        for (expected_amp, freq) in amps.iter().zip(freqs.iter()) {
            let amp = goertzel(&signal, *freq);
            assert!(
                (amp - expected_amp).abs() < 0.01,
                "Frequency {}: expected amplitude {}, got {}",
                freq,
                expected_amp,
                amp
            );
        }
    }

    #[test]
    fn test_batch_single() {
        // Generate test signal
        let n_samples = 512;
        let frequency = 1.0 / 128.0; // Normalized frequency
        let amplitude_expected = 2.5;
        let phase_expected = std::f64::consts::PI / 4.0;

        let signal = wave(amplitude_expected, frequency, phase_expected, n_samples);

        // Run Goertzel algorithm
        let amp = goertzel_batch(&signal, &vec![frequency]);

        // Check results (with tolerance for floating-point)
        assert!(
            (amp[0] - amplitude_expected).abs() < 1e-10,
            "Expected amplitude {}, got {}",
            amplitude_expected,
            amp[0]
        );
    }

    #[test]
    fn test_batch_processing_wave() {
        let n_samples = 1024;

        // Create composite signal
        let freqs = [1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0];
        let amps = [1.0, 2.0, 0.5, 0.25, 0.125];
        let mut signal = vec![0.0; n_samples];
        for (amp, freq) in amps.iter().zip(freqs.iter()) {
            let component = wave(*amp, *freq, 0.0, n_samples);
            for (i, val) in component.iter().enumerate() {
                signal[i] += val;
            }
        }

        let measured_amps = goertzel_batch(&signal, &freqs);

        // Test each frequency
        for ((expected_amp, freq), measured_amp) in amps.iter().zip(freqs.iter()).zip(measured_amps)
        {
            assert!(
                (measured_amp - expected_amp).abs() < 1e-10,
                "Frequency {}: expected amplitude {}, got {}",
                freq,
                expected_amp,
                measured_amp
            );
        }
    }

    #[test]
    fn test_batch_processing() {
        let n_samples = 2048;
        let signal: Vec<f64> = (0..n_samples)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0) // Random noise [-1, 1]
            .collect();

        let frequencies = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Batch processing
        let results = goertzel_batch(&signal, &frequencies);

        // Verify against individual calls
        for (i, freq) in frequencies.iter().enumerate() {
            let amp = goertzel(&signal, *freq);
            assert!(
                (results[i] - amp).abs() < 1e-10,
                "Batch result mismatch at index {} - Expected: {}, got: {}",
                i,
                amp,
                results[i]
            );
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test with zeros
        let signal = vec![0.0; 100];
        let amp = goertzel(&signal, 0.1);
        assert_eq!(amp, 0.0, "Zero signal should produce zero amplitude");

        // Test with single sample
        let signal = vec![1.0];
        let amp = goertzel(&signal, 0.0);
        assert!(
            amp.is_finite(),
            "Single sample should produce finite result"
        );

        // Test empty signal
        let signal: Vec<f64> = vec![];
        let amp = goertzel(&signal, 0.1);
        assert_eq!(amp, 0.0, "Empty signal should produce zero amplitude");
    }

    #[test]
    #[should_panic(expected = "frequency")]
    fn test_invalid_frequency_too_high() {
        let signal = vec![1.0, 2.0, 3.0];
        goertzel(&signal, 1.5); // Should panic - frequency > 0.5
    }

    #[test]
    #[should_panic(expected = "frequency")]
    fn test_invalid_frequency_negative() {
        let signal = vec![1.0, 2.0, 3.0];
        goertzel(&signal, -0.1); // Should panic - negative frequency
    }

    #[test]
    fn test_numerical_stability() {
        // Test with large signals
        let n_samples = 100_000;
        let signal: Vec<f64> = (0..n_samples)
            .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * 1e6)
            .collect();

        let amp = goertzel(&signal, 0.1);

        assert!(amp.is_finite(), "Result should be finite");
        assert!(!amp.is_nan(), "Result should not be NaN");
    }

    #[test]
    fn test_linearity() {
        // Test that doubling amplitude doubles the result
        let n_samples = 512;
        let frequency = 0.1;

        let signal1 = wave(1.0, frequency, 0.0, n_samples);
        let signal2 = wave(2.0, frequency, 0.0, n_samples);

        let amp1 = goertzel(&signal1, frequency);
        let amp2 = goertzel(&signal2, frequency);

        assert!(
            (amp2 - 2.0 * amp1).abs() < 1e-10,
            "Doubling amplitude should double magnitude"
        );
    }

    #[test]
    fn test_phase_independence() {
        // Same frequency/amplitude with different phases should yield same magnitude
        let n_samples = 512;
        let frequency = 0.125;
        let amplitude = 3.0;

        let signal1 = wave(amplitude, frequency, 0.0, n_samples);
        let signal2 = wave(amplitude, frequency, std::f64::consts::PI / 2.0, n_samples);
        let signal3 = wave(amplitude, frequency, std::f64::consts::PI, n_samples);

        let amp1 = goertzel(&signal1, frequency);
        let amp2 = goertzel(&signal2, frequency);
        let amp3 = goertzel(&signal3, frequency);

        assert!(
            (amp1 - amp2).abs() < 1e-10,
            "Phase shift should not affect magnitude"
        );
        assert!(
            (amp1 - amp3).abs() < 1e-10,
            "Phase shift should not affect magnitude"
        );
    }
}
