from g4beam import *
from scan import *
import time

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

start = time.time()
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

print(f"Time elapsed: {time.time() - start:.1f}")
