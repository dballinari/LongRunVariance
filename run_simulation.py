import numpy as np
from TimeSeriesStuff.estimators import newey_west
from TimeSeriesStuff.dgps import sim_ma_process, acf_ma_process

theta = np.array([1, 0.5, 0.3])
# Compute true long-run variance
acf = acf_ma_process(theta=np.array([1, 0.5, 0.3]), sigma=1)
long_run_var = acf[0] + 2*np.sum(acf[1:])

# Simulate data and estimate long-run variance
num_sim = 200
est_long_run_var = np.zeros(num_sim)
for i in range(num_sim):
    # Simulate data: MA(2)
    x = sim_ma_process(n=100000, theta=np.array([1, 0.5, 0.3]), sigma=1, seed=i)
    # Estimate long-run variance
    est_long_run_var[i] = newey_west(x)

print(f"True long-run variance: {long_run_var:.4f}")
print(f"Estimated long-run variance: {est_long_run_var.mean():.4f}")