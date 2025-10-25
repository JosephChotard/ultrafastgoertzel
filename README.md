# UltraFast Goertzel

Ultra-fast Goertzel algorithm implementation with SIMD optimization.

This workspace contains:
- **crates/ultrafastgoertzel**: Core Rust implementation with SIMD optimizations
- **bindings/python**: Python bindings using PyO3

## Features

- ⚡ SIMD-optimized for maximum performance
- 🔢 Single and batch frequency analysis
- 🐍 Python bindings available
- 📊 Benchmarked on signals up to 1M samples

## Rust Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ultrafastgoertzel = { path = "crates/ultrafastgoertzel" }
```

Example:

```rust
use ultrafastgoertzel::{goertzel, goertzel_batch};

fn main() {
    let signal: Vec<f64> = vec![/* your signal data */];
    
    // Single frequency
    let magnitude = goertzel(&signal, 0.1);
    
    // Multiple frequencies (more efficient)
    let frequencies = vec![0.1, 0.2, 0.3];
    let magnitudes = goertzel_batch(&signal, &frequencies);
}
```

## Python Usage

See [crates/python/README.md](crates/python/README.md) for Python installation and usage.

```python
import ultrafastgoertzel as ufg

signal = [...]  # Your signal data
magnitude = ufg.goertzel(signal, 0.1)
```


## Benchmarks

Run benchmarks for the core library:

```bash
cargo bench
```

## License

WTFPL
