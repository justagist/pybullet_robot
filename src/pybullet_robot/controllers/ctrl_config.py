import numpy as np

KP_P = np.asarray([2000., 2000., 2000.]) # 6000
KP_O = np.asarray([300., 300., 300.]) # 300
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
    'P_f': np.asarray([0.3,0.3,0.3]),
    'P_tor': np.asarray([0.3,0.3,0.3]),
    'I_f': np.asarray([3.,3.,3.]),
    'I_tor': np.asarray([3.,3.,3.]),
    'error_thresh': np.asarray([0.005, 0.005, 0.5, 0.5]),
    'start_err': np.asarray([200.,200., 200., 200.]),
    'ft_directions': [0,0,0,0,0,0],
    'windup_guard': [100.,100.,100.,100.,100.,100.],
    'null_stiffness': [000.]*7,
}
