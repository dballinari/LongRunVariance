import numpy as np
from TimeSeriesStuff.estimators import newey_west
from TimeSeriesStuff.dgps import sim_ma_process, acf_ma_process

# Simulate data: MA(2)
x = sim_ma_process(n=10000, theta=np.array([1, 0.5, 0.3]), sigma=1, seed=0)
# Compute true long-run variance
acf = acf_ma_process(theta=np.array([1, 0.5, 0.3]), sigma=1)
long_run_var = acf[0] + 2*np.sum(acf[1:])
# Estimate long-run variance
est_long_run_var = newey_west(x)