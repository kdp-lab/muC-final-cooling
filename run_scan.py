# File used for running scans with parallelization. I'll modify this for each scan I run.

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


def run_optimize_b(beta):
        pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=12000)
        def goal_fun(x):
            length, angle = x
            return emittances(cut_outliers(run_distribution(pre_w1, length, angle, vd_dist, axis=0, maxStep=1)))[0]
        res = scipy.optimize.minimize(goal_fun, [7.5, 45], method="Nelder-Mead", bounds=((1, 10), (30, 70)))
        return res

def run_optimize_a(alpha):
        pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=12000)
        def goal_fun(x):
            length, angle = x
            return emittances(cut_outliers(run_distribution(pre_w1, length, angle, vd_dist, axis=0, maxStep=1)))[0]
        res = scipy.optimize.minimize(goal_fun, [7.5, 45], method="Nelder-Mead", bounds=((1, 10), (30, 70)))
        return res


if __name__ == "__main__":
    # I'm no longer having the notebook import the scan ranges. These will have to be copied over.
    # I might eventually make the scans return objects bundling the ranges, which would make it easier.
    # particleCountRange = np.geomspace(1e3, 1e5, 10).astype(int)
    alphaRange = np.linspace(0, 1, 30)
    betaRange = np.linspace(0.25, 0.4, 30)

    # Current run: Scan over beta and alpha individually
    print("Starting scan over beta")
    results = run_scan(run_optimize_b, (betaRange,), "results/optimal geometry/beta.pkl", trials=30, processes=60)
    
    print("Starting scan over alpha")
    results = run_scan(run_optimize_a, (alphaRange,), "results/optimal geometry/alpha.pkl", trials=30, processes=60)