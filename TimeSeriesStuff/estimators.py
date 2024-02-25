import numpy as np

def newey_west(x: np.array, bandwidth: int=None):
    """
    Compute Newey-West standard errors for a given univariate time series.
    """
    x = np.array(x)
    n = len(x)
    x = x - x.mean()
    # reshape x to [n x 1] vector
    x = x.reshape(-1,1)
    # if no bandwidth is provided, use the automatic bandwidth selection
    if bandwidth is None:
        # TODO
        bandwidth = int(4 * (n / 100)**(2/9))
    # estimate variance
    V = np.dot(x.T, x) / n
    for i in range(1, bandwidth + 1):
        V += (1 - i / (bandwidth + 1)) * 2 * (np.dot(x[i:].T, x[:n-i]) / n )
    return V