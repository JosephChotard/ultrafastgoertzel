use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use ultrafastgoertzel as ufg;

/// Helper to extract f64 slice from either a Python list or NumPy array
fn extract_signal<'py>(_py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    // Try NumPy array first (more efficient)
    if let Ok(array) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(array.as_slice()?.to_vec());
    }

    // Fall back to Python list
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            result.push(item.extract::<f64>()?);
        }
        return Ok(result);
    }

    Err(PyValueError::new_err(
        "signal must be a list of floats or a NumPy array of float64",
    ))
}

/// Helper to extract f64 slice from either a Python list or NumPy array (for frequencies)
fn extract_frequencies<'py>(_py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    // Try NumPy array first (more efficient)
    if let Ok(array) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(array.as_slice()?.to_vec());
    }

    // Fall back to Python list
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            result.push(item.extract::<f64>()?);
        }
        return Ok(result);
    }

    Err(PyValueError::new_err(
        "frequencies must be a list of floats or a NumPy array of float64",
    ))
}

/// Compute the Goertzel algorithm for a single frequency.
///
/// Accepts both Python lists and NumPy arrays. When using NumPy arrays,
/// the GIL is released for better multi-threaded performance.
///
/// Args:
///     signal: Input signal as a list of floats or NumPy array of float64
///     frequency: Normalized frequency to analyze (must be in range [0.0, 0.5])
///
/// Returns:
///     float: Magnitude of the signal at the specified frequency
///
/// Raises:
///     ValueError: If frequency is not in the valid range [0.0, 0.5]
///
/// Example:
///     >>> import ultrafastgoertzel as ufg
///     >>> import numpy as np
///     >>> # Works with lists
///     >>> signal = [0.1, 0.2, 0.3, 0.4]
///     >>> magnitude = ufg.goertzel(signal, 0.1)
///     >>> # Also works with NumPy arrays (faster with threads!)
///     >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(1000))
///     >>> magnitude = ufg.goertzel(signal, 0.1)
#[pyfunction]
#[pyo3(text_signature = "(signal, frequency)")]
fn goertzel(py: Python<'_>, signal: &Bound<'_, PyAny>, frequency: f64) -> PyResult<f64> {
    if frequency < 0.0 || frequency > 0.5 {
        return Err(PyValueError::new_err(
            "frequency must be in range [0.0, 0.5]",
        ));
    }

    let signal_vec = extract_signal(py, signal)?;

    // Release the GIL while performing computation
    let result = py.allow_threads(|| ufg::goertzel(&signal_vec, frequency));

    Ok(result)
}

/// Compute the Goertzel algorithm for multiple frequencies in batch mode.
///
/// Accepts both Python lists and NumPy arrays. When using NumPy arrays,
/// the GIL is released for better multi-threaded performance. This is more
/// efficient than calling `goertzel()` multiple times for different frequencies
/// on the same signal, as it uses SIMD optimizations.
///
/// Args:
///     signal: Input signal as a list of floats or NumPy array of float64
///     frequencies: List/array of normalized frequencies to analyze (each must be in range [0.0, 0.5])
///
/// Returns:
///     numpy.ndarray: NumPy array of magnitudes for each frequency
///
/// Raises:
///     ValueError: If any frequency is not in the valid range [0.0, 0.5]
///
/// Example:
///     >>> import ultrafastgoertzel as ufg
///     >>> import numpy as np
///     >>> # Works with lists
///     >>> signal = [0.1, 0.2, 0.3, 0.4]
///     >>> magnitudes = ufg.goertzel_batch(signal, [0.1, 0.2])
///     >>> # Works with NumPy arrays (faster with threads!)
///     >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(1000))
///     >>> frequencies = np.array([0.1, 0.2, 0.3])
///     >>> magnitudes = ufg.goertzel_batch(signal, frequencies)
#[pyfunction]
#[pyo3(text_signature = "(signal, frequencies)")]
fn goertzel_batch<'py>(
    py: Python<'py>,
    signal: &Bound<'_, PyAny>,
    frequencies: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal_vec = extract_signal(py, signal)?;
    let frequencies_vec = extract_frequencies(py, frequencies)?;

    // Validate frequencies
    for &freq in &frequencies_vec {
        if freq < 0.0 || freq > 0.5 {
            return Err(PyValueError::new_err(
                "all frequencies must be in range [0.0, 0.5]",
            ));
        }
    }

    // Release the GIL while performing computation
    let result = py.allow_threads(|| ufg::goertzel_batch(&signal_vec, &frequencies_vec));

    // Convert result to numpy array
    Ok(PyArray1::from_vec(py, result))
}

/// Ultra-fast Goertzel algorithm implementation with SIMD optimization.
///
/// This module provides a high-performance implementation of the Goertzel algorithm
/// for computing the magnitude of a signal at specific frequencies. It uses SIMD
/// (Single Instruction, Multiple Data) instructions for efficient batch processing.
///
/// The Goertzel algorithm is particularly useful for detecting specific frequency
/// components in a signal without computing a full FFT.
///
/// Functions:
///     goertzel: Compute magnitude for a single frequency (accepts lists or NumPy arrays)
///     goertzel_batch: Compute magnitudes for multiple frequencies (accepts lists or NumPy arrays, recommended)
///
/// Note:
///     Frequencies are normalized, where 0.5 represents the Nyquist frequency
///     (half the sampling rate). For example, if your sampling rate is 1000 Hz,
///     a normalized frequency of 0.1 corresponds to 100 Hz.
///
///     Both functions automatically detect NumPy arrays and release Python's GIL
///     for true multi-threaded parallelism when processing them.
#[pymodule]
fn _ultrafastgoertzel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(goertzel, m)?)?;
    m.add_function(wrap_pyfunction!(goertzel_batch, m)?)?;
    Ok(())
}
