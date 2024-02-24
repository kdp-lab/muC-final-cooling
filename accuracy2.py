# Additional simulation accuracy scans, updated to work with the new parallelization setup

# Import necessary packages
from g4beam import *
from scan import *

import numpy as np
import json

# Parameter list
PARAMS = [
    "t_emit",
    "momentum",
    "beta",
    "alpha",
    "l_emit",
    "pz_std",
    "vd_dist",
    "w1_length",
    "w1_angle",
    "w2_length",
    "w2_angle",
    "drift_length",
    "rf_freq",
    "rf_phase",
    "rf_length",
    "rf_grad"
]

# Load file
with open("results/parameters/145_new.json", "rb+") as file:
    parameters = json.load(file)
    globals().update(parameters)


def run_optimize(N):
        start_time = time.time()
        pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=N)
        def goal_fun(x):
            length, angle = x
            return emittances(cut_outliers(run_distribution(pre_w1, length, angle, vd_dist, axis=0, maxStep=1)))[0]
        res = scipy.optimize.minimize(goal_fun, [7.5, 45], method="Nelder-Mead", bounds=((1, 10), (30, 70)))
        return res, time.time()-start_time


def run_optimize_v2(N):
        start_time = time.time()
        def goal_fun(x):
            length, angle = x
            pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=N)
            return emittances(cut_outliers(run_distribution(pre_w1, length, angle, vd_dist, axis=0, maxStep=1)))[0]
        res = scipy.optimize.minimize(goal_fun, [7.5, 45], method="Nelder-Mead", bounds=((1, 10), (30, 70)))
        return res, time.time()-start_time


if __name__ == "__main__":
    # I'm no longer having the notebook import the scan ranges. These will have to be copied over.
    # I might eventually make the scans return objects bundling the ranges, which would make it easier.
    particleCountRange = np.geomspace(1e3, 1e5, 10).astype(int)

    # Repeat the optimal v.s. N scan from last time to check that I get the same results
    print("Starting optimal v.s. N scan")
    results = run_scan(run_optimize, (particleCountRange,), "results/accuracy_optim_particleCount_rerun.pkl", trials=30, processes=32)