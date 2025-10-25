# ultrafastgoertzel Python Bindings

Ultra-fast Goertzel algorithm implementation with SIMD optimization for Python.
## Usage

```python
import ultrafastgoertzel as ufg
import math

# Generate a test signal (sine wave at frequency 0.1)
signal = [math.sin(2 * math.pi * 0.1 * i) for i in range(1000)]

# Analyze a single frequency
magnitude = ufg.goertzel(signal, 0.1)
print(f"Magnitude at 0.1: {magnitude:.4f}")

# Analyze multiple frequencies efficiently (recommended)
frequencies = [0.1, 0.2, 0.3]
magnitudes = ufg.goertzel_batch(signal, frequencies)
for freq, mag in zip(frequencies, magnitudes):
    print(f"Frequency {freq}: {mag:.4f}")
```

## Frequency Normalization

Frequencies are normalized, where:
- 0.0 = DC (0 Hz)
- 0.5 = Nyquist frequency (half the sampling rate)

For example, if your sampling rate is 1000 Hz:
- 0.1 represents 100 Hz
- 0.25 represents 250 Hz
- 0.5 represents 500 Hz (Nyquist)

## Performance

This implementation uses SIMD instructions for optimal performance. The `goertzel_batch` function is particularly efficient when analyzing multiple frequencies on the same signal, as it can process multiple frequencies in parallel.

## License

WTFPL License.