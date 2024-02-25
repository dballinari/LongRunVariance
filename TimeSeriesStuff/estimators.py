import numpy as np

def newey_west(x: np.array, bandwidth: int=None):
    """
    Compute Newey-West standard errors for a given univariate time series.
    """
    x = np.array(x)
    nobs = len(x)
    # work with demeaned x to avoid the need to estimate the mean
    x = x - x.mean()
    # reshape x to [n x 1] vector
    x = x.reshape(-1,1)
    # if no bandwidth is provided, use the automatic bandwidth selection
    if bandwidth is None:
        s1 = 0
        s0 = 0
        n = int(4 * (nobs / 100)**(2/9))
        for j in range(n+1):
            sigma_j = np.dot(x[j:].T, x[:nobs-j])/nobs
            s1 += 2 * j * sigma_j
            s0 += (1 + j!=0) * sigma_j
        bandwidth = int( 1.1447 * (nobs * (s1/s0)**2) **(1/3))
        
    # estimate variance
    V = np.dot(x.T, x) / nobs
    for i in range(1, bandwidth + 1):
        V += (1 - i / (bandwidth + 1)) * 2 * (np.dot(x[i:].T, x[:nobs-i]) / nobs )
    return V