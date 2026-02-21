from numpy import (
    array,
    convolve,
    ones
)


def mov_mean(x: array, window_len: int, mode: str = 'same') -> array:
    """Compute the moving average of a signal using convolution.

    This function applies a moving average filter to an input signal by convolving
    it with a normalized window of ones. It smooths the signal by averaging values
    within a sliding window.

    Args:
        x (array): Input signal to be filtered.
        window_len (int): Length of the moving average window in samples.
        mode (str, optional): Convolution mode. Options are 'full', 'same', or 'valid'.
            'same' returns output of same length as input. Defaults to 'same'.

    Returns:
        array: Smoothed output signal after convolution, with shape depending on mode parameter.
    """
    return convolve(x, ones(window_len) / window_len, mode)
