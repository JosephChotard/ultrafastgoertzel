import numpy as np
import pytest
from typing import Sequence
import ultrafastgoertzel as ufg


# --- Helper: Generate sine wave ---
def sine_wave(amplitude: float, frequency: float, phase: float, n_samples: int) -> np.ndarray:
    """Generate a pure sine wave of given length."""
    t = np.arange(n_samples)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


# --------------------------------------------------------------------------- #
#                        SINGLE FREQUENCY TESTS
# --------------------------------------------------------------------------- #

def test_pure_sine_wave_detection():
    n_samples = 512
    freq = 1.0 / 128  # 0.0078125 → well within [0, 0.5]
    expected_amp = 3.7

    signal = sine_wave(expected_amp, freq, np.pi / 3, n_samples)
    detected_amp = ufg.goertzel(signal, freq)

    assert np.isclose(detected_amp, expected_amp, atol=1e-12)


def test_multiple_frequency_isolation():
    n_samples = 1024
    components = [
        (1.5, 1.0 / 256, 0.0),
        (2.8, 1.0 / 128, np.pi / 5),
        (0.9, 1.0 / 64,  -np.pi / 6),
    ]

    signal = sum(sine_wave(amp, freq, phase, n_samples) for amp, freq, phase in components)

    for expected_amp, freq, _ in components:
        detected_amp = ufg.goertzel(signal.astype(np.float64), freq)
        assert np.isclose(detected_amp, expected_amp, atol=0.12)


def test_zero_signal_returns_zero():
    signal = np.zeros(200)
    amp = ufg.goertzel(signal, 0.2)
    assert amp == 0.0


def test_empty_signal_returns_zero():
    signal = np.array([])
    amp = ufg.goertzel(signal, 0.1)
    assert amp == 0.0


def test_linearity_scaling():
    n_samples = 512
    freq = 0.1
    signal1 = sine_wave(1.0, freq, 0.0, n_samples)
    signal2 = sine_wave(2.0, freq, 0.0, n_samples)

    amp1 = ufg.goertzel(signal1, freq)
    amp2 = ufg.goertzel(signal2, freq)

    assert np.isclose(amp2, 2.0 * amp1, atol=1e-12)


def test_phase_independence():
    n_samples = 512
    freq = 0.125
    amp = 3.0
    phases = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    magnitudes = [
        ufg.goertzel(sine_wave(amp, freq, ph, n_samples), freq)
        for ph in phases
    ]

    assert all(np.isclose(m, magnitudes[0], atol=1e-12) for m in magnitudes)


def test_numerical_stability_with_large_amplitudes():
    n = 100_000
    noise = np.random.randn(n) * 1e8
    tone = 1e7 * np.sin(2 * np.pi * 0.1 * np.arange(n))
    signal = noise + tone

    amp = ufg.goertzel(signal, 0.1)

    assert np.isfinite(amp)
    assert amp > 1e6  # strong tone should dominate


# --------------------------------------------------------------------------- #
#                           BATCH PROCESSING TESTS
# --------------------------------------------------------------------------- #

def test_batch_processing_consistency():
    signal = np.random.randn(600) * 0.5 + np.sin(2 * np.pi * 0.15 * np.arange(600))
    frequencies = np.array([0.05, 0.15, 0.25, 0.35])

    batch_amps = ufg.goertzel_batch(signal, frequencies)

    for freq, batch_amp in zip(frequencies, batch_amps):
        single_amp = ufg.goertzel(signal, freq)
        assert np.isclose(single_amp, batch_amp, atol=1e-12)


def test_batch_single_frequency():
    n_samples = 512
    freq = 1.0 / 128
    signal = sine_wave(2.5, freq, np.pi / 4, n_samples)

    single_amp = ufg.goertzel(signal, freq)
    batch_amp = ufg.goertzel_batch(signal, [freq])[0]

    assert np.isclose(single_amp, batch_amp, atol=1e-12)


def test_batch_with_list_inputs():
    signal = [0.1, 0.2, 0.3, 0.4, 0.5]
    freqs = [0.1, 0.2]

    amps = ufg.goertzel_batch(signal, freqs)
    assert len(amps) == 2
    assert all(isinstance(a, float) for a in amps)


# --------------------------------------------------------------------------- #
#                           ERROR HANDLING
# --------------------------------------------------------------------------- #

def test_invalid_frequency_too_high():
    signal = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="frequency.*0\\.0.*0\\.5"):
        ufg.goertzel(signal, 0.6)


def test_invalid_frequency_negative():
    signal = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="frequency.*0\\.0.*0\\.5"):
        ufg.goertzel(signal, -0.1)


def test_batch_invalid_frequency():
    signal = np.random.randn(100)
    freqs = [0.1, 0.6, 0.3]
    with pytest.raises(ValueError):
        ufg.goertzel_batch(signal, freqs)