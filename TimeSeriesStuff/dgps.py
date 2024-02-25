import numpy as np


def sim_ma_process(n: int, theta: np.array, sigma: float, seed: int=None):
    """
    Simulate an MA process of order q.

    Args:
    n: int
        Length of the time series to simulate.
    theta: np.array
        Coefficients of the MA process. The first element should be 1.
    sigma: float
        Standard deviation of the white noise.
    seed: int
        Seed for the random number generator.

    Returns:
    np.array
        Simulated time series.
    """
    if seed is not None:
        np.random.seed(seed)
    q = len(theta) - 1
    e = sigma * np.random.normal(size=n + q)
    x = np.convolve(e, theta, mode='valid')
    return x[q:]


def acf_ma_process(theta: np.array, sigma: float, corr: bool=False):
    """
    Compute the theoretical autocorrelation function of an MA process.

    Args:
    theta: np.array
        Coefficients of the MA process. The first element should be 1.
    sigma: float
        Standard deviation of the white noise.

    Returns:
    np.array
        Autocorrelation function of the MA process.
    """
    q = len(theta) - 1
    acf = np.zeros(q + 1)
    acf[0] = 1
    for k in range(0, q + 1):
        acf[k] = np.sum(theta[:q - k + 1] * theta[k:])
    acf *= sigma**2
    if corr:
        acf /= acf[0]
    return acf