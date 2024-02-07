'''
Author: Anthony Badea
Date: 02/01/2024
'''

from g4beam import *
from scan import *
import time
from skopt import gp_minimize
import numpy as np

def f(x):
    params = {
        't_emit': 0.145, # mm
        'momentum': 100, # MeV/c
        'beta': 0.03, # m
        'alpha': 1, # dimensionless
        'l_emit': 1, # mm
        'pz_std': 1, # MeV/c
        'vd_dist': 24, # mm
        'w1_length': x[0],
        'w1_angle': x[1],
        'w2_length': 6.724887901827298,
        'w2_angle': 42.245718529695516,
        'drift_length': 16000,
        'rf_freq': 0.025,
        'rf_phase': 0.001987066319906211,
        'rf_length': 5153.756925848655,
        'rf_grad': 4.046563465382562
    }
    x,y,z = run(params)
    return z

def run(params):
    # Run best case
    pre_w1 = gen_distribution(
        (params["beta"], params["alpha"], params["t_emit"], 0, 0),
        (params["beta"], params["alpha"], params["t_emit"], 0, 0),
        params["momentum"],
        params["pz_std"],
        z_emit=params["l_emit"],
        N=50000)
    pre_w1["PDGid"] = -13
    print("Running first wedge")
    post_w1 = run_distribution(
        pre_w1,
        params["w1_length"],
        params["w1_angle"],
        params["vd_dist"],
        axis=0)
    # post_correct = post_w1
    post_correct = remove_dispersion(post_w1)
    reverse_transverse = post_correct.copy(deep=True)
    reverse_transverse["Px"] *= -1
    reverse_transverse["Py"] *= -1
    drift_to_start = params["drift_length"]-params["rf_length"]/2
    post_drift = recenter_t(z_prop(post_correct, drift_to_start))
    no_transverse = remove_transverse(post_drift)
    print("Running RF cavity")
    post_cavity = cut_pz(recenter_t(run_g4beam(no_transverse, "./configs/G4_RFCavity.g4bl", RF_length=params["rf_length"], frfcool=params["rf_freq"], ficool=params["rf_phase"], Vrfcool=params["rf_grad"], nparticles=len(no_transverse))), tails=0.15)
    pre_w2 = recombine_transverse(post_cavity, reverse_transverse)
    print("Running second wedge")
    post_w2 = run_distribution(
        pre_w2,
        params["w2_length"],
        params["w2_angle"],
        params["vd_dist"],
        axis=1
    )
    post_w2_cut = recenter_t(cut_outliers(post_w2))
    print_all_params(post_w2_cut)

    # get distributions
    # emits = {}
    # SAMPLE_DISTS = [pre_w1, post_correct, pre_w2, post_w2, post_w2_cut]
    # SAMPLE_TITLES = ["Initial distribution", "After first wedge", "After RF cavity + 15% cut", "After second wedge", "After 4 sigma cut"]
    # for sample, dist in zip(SAMPLE_TITLES, SAMPLE_DISTS):
    #     x_emit, y_emit, z_emit = emittances(cut_outliers(run_distribution(dist, params["w1_length"], params["w1_angle"], params["vd_dist"], axis=0)))
    #     print(f"{sample}: ", x_emit, y_emit, z_emit)
    #     emits[sample] = [x_emit, y_emit, z_emit]
    # return emits

    return emittances(cut_outliers(run_distribution(post_w2_cut, params["w1_length"], params["w1_angle"], params["vd_dist"], axis=0)))

if __name__ == "__main__":
    # list of parameters to optimize
    params = {
        't_emit': 0.145, # mm
        'momentum': 100, # MeV/c
        'beta': 0.03, # m
        'alpha': 1, # dimensionless
        'l_emit': 1, # mm
        'pz_std': 1, # MeV/c
        'vd_dist': 24, # mm
        'w1_length': 9.20751061747799,
        'w1_angle': 49.78231333988419,
        'w2_length': 6.724887901827298,
        'w2_angle': 42.245718529695516,
        'drift_length': 16000,
        'rf_freq': 0.025,
        'rf_phase': 0.001987066319906211,
        'rf_length': 5153.756925848655,
        'rf_grad': 4.046563465382562
    }
    
    
    # example of full run through
    # start = time.time()
    # emits = run(params)
    # print(f"Time elapsed: {time.time() - start:.1f}")
    # print(emits)
    np.int = int
    res = gp_minimize(f,                  # the function to minimize
                      [
                        (1, 10),          # bounds on 1st wedge length
                        (30, 70)          # bounds on 1st wedge angle
                      ],
                      x0=[7.5, 45],       # starting values
                      acq_func="EI",      # the acquisition function
                      n_calls=15,         # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      # noise=0.1**2,       # the noise level (optional)
                      random_state=1234)   # the random seed
    
    print("x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun))
    print(res)
