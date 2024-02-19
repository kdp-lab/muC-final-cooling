# Scans for determining the effect on simulation accuracy of max steps and particle counts

# Import necessary packages
from g4beam import *
from scan import *

import numpy as np
import pandas as pd
from tqdm import *
import pickle
import lzma

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

# Helper function to load parameter sets
def load_params(filename):
    with open(filename, "rb+") as file:
        parameters = pickle.load(file)
        globals().update(parameters)
        print("Loaded from", filename)

# Set scan ranges. Scan ranges are set outside the main method so they can be loaded from the notebook
particleCountRange = np.geomspace(1e4, 1e6, 10).astype(int)
maxStepRange = np.geomspace(0.01, 10, 10)
wedgeStepRange = np.geomspace(0.01, 10, 10)
particleCountOptimRange = np.geomspace(1e4, 1e5, 10).astype(int)


if __name__ == "__main__":
    # Load up the 145 μm optimized parameters
    load_params("results/parameters/145_new.pkl")

    # # Run scan for particle count
    # print("Running particle count scan")
    # def fun(N):
    #     pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=N)
    #     pre_w1["PDGid"] = -13
    #     start_time = time.time()
    #     post_w1 = run_distribution(pre_w1, w1_length, w1_angle, vd_dist, axis=0)
    #     return post_w1, time.time()-start_time
    
    # results = run_scan(fun, (particleCountRange,), "results/accuracy_particleCount.pkl.lzma", trials=30)

    # # Run scan for max step
    # print("Running max step scan")
    # def fun(maxStep):
    #     pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=20000)
    #     pre_w1["PDGid"] = -13
    #     start_time = time.time()
    #     post_w1 = run_distribution(pre_w1, w1_length, w1_angle, vd_dist, axis=0, maxStep=maxStep)
    #     return post_w1, time.time()-start_time
    
    # results = run_scan(fun, (maxStepRange,), "results/accuracy_maxStep.pkl.lzma", trials=30)

    # # Run scan for wedge max step
    # print("Running max step in wedge scan")
    # def fun(maxStepInWedge):
    #     pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=20000)
    #     pre_w1["PDGid"] = -13
    #     start_time = time.time()
    #     post_w1 = run_distribution(pre_w1, w1_length, w1_angle, vd_dist, axis=0, maxStepInWedge=maxStepInWedge)
    #     return post_w1, time.time()-start_time
    
    # results = run_scan(fun, (wedgeStepRange,), "results/accuracy_wedgeStep.pkl.lzma", trials=30)

    # Run scan on optimization accuracy with increasing particle count
    def fun(N):
        start_time = time.time() # Starting it here as the time to generate the distribution should probably be counted
        pre_w1 = gen_distribution((beta, alpha, t_emit, 0, 0), (beta, alpha, t_emit, 0, 0), momentum, pz_std, z_emit=l_emit, N=N)
        def goal_fun(x):
            length, angle = x
            return emittances(cut_outliers(run_distribution(pre_w1, length, angle, vd_dist, axis=0, maxStep=1)))[0]
        res = scipy.optimize.minimize(goal_fun, [7.5, 45], method="Nelder-Mead", bounds=((1, 10), (30, 70)))
        return res.x[0], res.x[1], res.fun, time.time()-start_time
    results = run_scan(fun, (particleCountOptimRange,), "results/accuracy_optim_particleCount.pkl.lzma", trials=30)