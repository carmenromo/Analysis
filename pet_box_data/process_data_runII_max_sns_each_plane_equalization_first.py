import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

from antea.reco import petit_reco_functions as prf

import data_taking_petalo_functions as pf


print(datetime.datetime.now())

arguments = pf.parse_args_tofpets(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
run_no    = arguments.run_no
tofpet_d  = arguments.tofpet_d
tofpet_c  = arguments.tofpet_c
out_path  = arguments.out_path


def compute_max_sns_per_plane(df, variable='efine_corrected', det_plane=True):
    if det_plane:
        df = df[df.tofpet_id == tofpet_d]
    else:
        df = df[df.tofpet_id == tofpet_c]
    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id #channel_id

def compute_tmin_per_plane(df, det_plane=True):
    if det_plane:
        df = df[df.tofpet_id == tofpet_d]
    else:
        df = df[df.tofpet_id == tofpet_c]
    return df.t.min()

norm_s_id_R12334 = {11: 316.598, 12: 333.760, 13: 348.095, 14: 304.341, 15: 315.501, 16: 305.090, 17: 270.018, 18: 250.164,
                    21: 315.064, 22: 327.413, 23: 289.244, 24: 284.921, 25: 313.378, 26: 301.873, 27: 258.367, 28: 241.835,
                    31: 317.901, 32: 321.779, 33: 316.800, 34: 285.201, 35: 336.234, 36: 300.655, 37: 257.138, 38: 263.906,
                    41: 264.397, 42: 261.548, 43: 289.330, 44: 292.119, 45: 270.566, 46: 310.494, 47: 270.508, 48: 274.084,
                    51: 317.009, 52: 323.511, 53: 317.707, 54: 261.675, 55: 281.559, 56: 261.741, 57: 291.251, 58: 270.885,
                    61: 343.809, 62: 288.743, 63: 312.118, 64: 312.352, 65: 290.119, 66: 0, 67: 265.882, 68: 0,
                    71: 298.913, 72: 299.055, 73: 317.224, 74: 264.760, 75: 263.110, 76: 297.701, 77: 279.510, 78: 248.714,
                    81: 327.664, 82: 287.615, 83: 266.467, 84: 0, 85: 249.069, 86: 267.882, 87: 227.026, 88: 231.074,
                    111: 386.532, 112: 378.577, 113: 433.585, 114: 414.029, 115: 338.911, 116: 0, 117: 376.256, 118: 355.226,
                    121: 449.234, 122: 399.509, 123: 0, 124: 370.646, 125: 409.107, 126: 369.666, 127: 386.464, 128: 360.790,
                    131: 428.003, 132: 416.891, 133: 0, 134: 291.589, 135: 376.101, 136: 383.951, 137: 362.155, 138: 408.116,
                    141: 348.738, 142: 399.910, 143: 407.306, 144: 415.319, 145: 338.106, 146: 372.679, 147: 374.699, 148: 374.272,
                    151: 419.248, 152: 420.372, 153: 414.236, 154: 374.020, 155: 378.266, 156: 395.739, 157: 354.413, 158: 377.962,
                    161: 427.713, 162: 422.081, 163: 393.928, 164: 449.194, 165: 403.344, 166: 363.387, 167: 359.035, 168: 394.252,
                    171: 358.108, 172: 387.687, 173: 419.885, 174: 400.831, 175: 361.645, 176: 390.571, 177: 382.550, 178: 345.825,
                    181: 420.301, 182: 389.274, 183: 380.415, 184: 385.448, 185: 343.981, 186: 373.009, 187: 383.120, 188: 405.166}

def apply_norm_s_id_R12334(sid: int) -> float:
    if sid < 100:
        norm = 275
    else:
        norm = 370
    return norm_s_id_R12334[sid]/norm

for i in range(start, start+numb):
    df0 = pd.DataFrame({})
    f = f'/analysis/{run_no}/hdf5/proc/linear_interp/files/run_{run_no}_{i:04}_trigger1_waveforms.h5'
    try:
        store = pd.HDFStore(f, 'r')
    except OSError:
        print(f'Error with file {f}')
        continue
    for key in store.keys():
        df = store.get(key)
        df = df.drop(columns=['ct_data', 'ctdaq', 'tac_id', 'ecoarse', 'tfine', 'tcoarse_extended', 'tfine_corrected']) #Remove unused columns
        df = df[df.cluster != -1] ## Filtering events with only one sensor

        df_coinc  = prf.compute_coincidences(df, evt_groupby=['evt_number', 'cluster'])

        df_coinc['efine_norm'] = df_coinc['efine_corrected']/df_coinc['sensor_id'].apply(apply_norm_s_id_R12334)

        max_sns_all0 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_max_sns_per_plane, variable='efine_norm', det_plane=True)
        max_sns_all2 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_max_sns_per_plane, variable='efine_norm', det_plane=False)
        df_coinc['max_sns0'] = max_sns_all0[df_coinc.index].values
        df_coinc['max_sns2'] = max_sns_all2[df_coinc.index].values

        tmin_all0 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_tmin_per_plane, det_plane=True)
        tmin_all2 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_tmin_per_plane, det_plane=False)
        df_coinc['tmin0'] = tmin_all0[df_coinc.index].values
        df_coinc['tmin2'] = tmin_all2[df_coinc.index].values

        df0 = pd.concat([df0, df_coinc], ignore_index=False, sort=False)

    out_file  = f'{out_path}/data_coinc_runII_ch_max_sns_equaliz_first_tmin_R{run_no}_{i}_{i_key}.h5'

    df    = df0.reset_index()
    store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

print(datetime.datetime.now())
