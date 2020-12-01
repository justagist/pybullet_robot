import numpy as np

KP_P = np.asarray([5000., 5000., 5000.])
KP_O = np.asarray([500., 500., 500.])
OSImpConfig = {
    'P_pos': KP_P,
    'D_pos': 2*np.sqrt(KP_P),
    'P_ori': KP_O,
    'D_ori': np.asarray([0.01,0.01,0.01]),
    'rate': 100,
    'error_thresh':0.005,
    'start_err': 200.
}
