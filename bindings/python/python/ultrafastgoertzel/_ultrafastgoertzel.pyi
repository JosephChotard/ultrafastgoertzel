"""
Ultra-fast Goertzel algorithm implementation with SIMD optimization.

This module provides a high-performance implementation of the Goertzel algorithm
for computing the magnitude of a signal at specific frequencies. It uses SIMD
(Single Instruction, Multiple Data) instructions for efficient batch processing.

The Goertzel algorithm is particularly useful for detecting specific frequency
components in a signal without computing a full FFT.

Note:
    Frequencies are normalized, where 0.5 represents the Nyquist frequency
    (half the sampling rate). For example, if your sampling rate is 1000 Hz,
    a normalized frequency of 0.1 corresponds to 100 Hz.
    
    Both functions automatically detect NumPy arrays and release Python's GIL 
    for true multi-threaded parallelism when processing them.
"""

from typing import Sequence
import numpy as np
import numpy.typing as npt

def goertzel(
    signal: Sequence[float] | npt.NDArray[np.float64], 
    frequency: float
) -> float:
    """
    Compute the Goertzel algorithm for a single frequency.
    
    Accepts both Python lists and NumPy arrays. When using NumPy arrays,
    the GIL is released for better multi-threaded performance.

    Args:
        signal: Input signal as a sequence of floats or NumPy array of float64
        frequency: Normalized frequency to analyze (must be in range [0.0, 0.5])

    Returns:
        Magnitude of the signal at the specified frequency

    Raises:
        ValueError: If frequency is not in the valid range [0.0, 0.5]

    Example:
        >>> import ultrafastgoertzel as ufg
        >>> import numpy as np
        >>> # Works with lists
        >>> signal = [0.1, 0.2, 0.3, 0.4]
        >>> magnitude = ufg.goertzel(signal, 0.1)
        >>> # Also works with NumPy arrays (faster with threads!)
        >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(1000))
        >>> magnitude = ufg.goertzel(signal, 0.1)
    """
    ...

def goertzel_batch(
    signal: Sequence[float] | npt.NDArray[np.float64], 
    frequencies: Sequence[float] | npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the Goertzel algorithm for multiple frequencies in batch mode.

    Accepts both Python lists and NumPy arrays. When using NumPy arrays,
    the GIL is released for better multi-threaded performance. This is more 
    efficient than calling `goertzel()` multiple times for different frequencies 
    on the same signal, as it uses SIMD optimizations.

    Args:
        signal: Input signal as a sequence of floats or NumPy array of float64
        frequencies: Sequence/array of normalized frequencies to analyze 
                    (each must be in range [0.0, 0.5])

    Returns:
        NumPy array of magnitudes for each frequency

    Raises:
        ValueError: If any frequency is not in the valid range [0.0, 0.5]

    Example:
        >>> import ultrafastgoertzel as ufg
        >>> import numpy as np
        >>> # Works with lists
        >>> signal = [0.1, 0.2, 0.3, 0.4]
        >>> magnitudes = ufg.goertzel_batch(signal, [0.1, 0.2])
        >>> # Works with NumPy arrays (faster with threads!)
        >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(1000))
        >>> frequencies = np.array([0.1, 0.2, 0.3])
        >>> magnitudes = ufg.goertzel_batch(signal, frequencies)
    """
    ...
