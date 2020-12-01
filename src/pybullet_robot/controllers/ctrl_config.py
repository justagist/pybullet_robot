import numpy as np

KP_P = np.asarray([5000., 5000., 5000.])
KP_O = np.asarray([500., 500., 500.])
OSImpConfig = {
    'P_pos': KP_P,
    'D_pos': 2*np.sqrt(KP_P),
    'P_ori': KP_O,
    'D_ori': np.asarray([0.01,0.01,0.01]),
    'error_thresh': np.asarray([0.005, 0.005]),
    'start_err': np.asarray([200., 200.])
}

OSHybConfig = {
    'P_pos': KP_P,
    'D_pos': 2*np.sqrt(KP_P),
    'P_ori': KP_O,
    'D_ori': np.asarray([0.01, 0.01, 0.01]),
    'P_f': np.asarray([0.6,0.6,0.6]),
    'P_tor': np.asarray([0.6,0.6,0.6]),
    'I_f': np.asarray([3.,3.,3.]),
    'I_tor': np.asarray([3.,3.,3.]),
    'error_thresh': np.asarray([0.005, 0.005, 0.5, 0.5]),
    'start_err': np.asarray([200.,200., 200., 200.]),
    'ft_directions': [0,0,0,0,0,0],
    'windup_guard': [100.,100.,100.,100.,100.,100.]
}
