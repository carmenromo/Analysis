import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

from antea.reco import petit_reco_functions as prf

import data_taking_petalo_functions as pf


print(datetime.datetime.now())

arguments = pf.parse_args_n_keys(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
i_key     = arguments.i_key
n_key     = arguments.n_key
run_no    = arguments.run_no
out_path  = arguments.out_path


def compute_max_sns_per_plane(df, variable='efine_corrected', det_plane=True):
    if det_plane:
        df = df[df.tofpet_id == 0]
    else:
        df = df[df.tofpet_id == 2]
    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id

def compute_max_sns_per_plane_new(df, variable='efine_corrected', det_plane=True):
    if det_plane:
        df = df[df.tofpet_id == 5]
    else:
        df = df[df.tofpet_id == 1]
    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id #channel_id

def compute_tmin_per_plane(df, det_plane=True):
    if det_plane:
        df = df[df.tofpet_id == 5]
    else:
        df = df[df.tofpet_id == 1]
    return df.t.min()

norm_s_id_R12252 = {11: 296.84, 12: 328.07, 13: 343.80, 14: 296.74, 15: 310.26, 16: 300.42, 17: 263.87, 18: 243.81,
                    21: 305.91, 22: 320.18, 23: 283.32, 24: 274.68, 25: 308.74, 26: 295.25, 27: 245.10, 28: 234.98,
                    31: 306.68, 32: 317.62, 33: 310.00, 34: 274.47, 35: 330.59, 36: 296.29, 37: 250.48, 38: 255.84,
                    41: 257.37, 42: 257.35, 43: 281.97, 44: 282.95, 45: 264.80, 46: 308.09, 47: 264.07, 48: 264.50,
                    51: 307.57, 52: 316.43, 53: 309.10, 54: 253.97, 55: 272.36, 56: 254.37, 57: 283.39, 58: 259.93,
                    61: 335.63, 62: 284.47, 63: 304.28, 64: 303.28, 65: 281.11, 66: 0, 67: 257.55, 68: 0,
                    71: 289.70, 72: 293.01, 73: 308.42, 74: 255.20, 75: 253.76, 76: 288.86, 77: 272.93, 78: 239.12,
                    81: 315.10, 82: 275.35, 83: 255.86, 84: 0, 85: 231.90, 86: 252.34, 87: 216.06, 88: 211.44,
                    111: 417.34, 112: 414.76, 113: 469.86, 114: 446.77, 115: 365.12, 116: 0, 117: 403.62, 118: 382.35,
                    121: 486.60, 122: 433.19, 123: 0, 124: 402.31, 125: 437.37, 126: 394.12, 127: 415.41, 128: 389.08,
                    131: 462.64, 132: 454.70, 133: 293.70, 134: 312.00, 135: 409.15, 136: 419.69, 137: 388.64, 138: 441.58,
                    141: 377.47, 142: 434.74, 143: 438.54, 144: 454.42, 145: 362.14, 146: 401.65, 147: 399.21, 148: 398.35,
                    151: 449.56, 152: 453.21, 153: 451.42, 154: 406.24, 155: 413.24, 156: 428.80, 157: 378.69, 158: 405.65,
                    161: 467.16, 162: 457.53, 163: 427.15, 164: 487.95, 165: 438.37, 166: 397.13, 167: 391.00, 168: 425.57,
                    171: 387.22, 172: 421.74, 173: 452.17, 174: 437.91, 175: 387.44, 176: 427.70, 177: 416.61, 178: 369.92,
                    181: 452.64, 182: 420.27, 183: 412.27, 184: 416.74, 185: 367.68, 186: 399.08, 187: 413.27, 188: 433.34}

def apply_norm_s_id_R12252(sid: int) -> float:
    if sid < 100:
        norm = 270
    else:
        norm = 415
    return norm_s_id_R12252[sid]/norm

norm_s_id_R12212 = {11: 300.53, 12: 328.69, 13: 340.54, 14: 298.74, 15: 312.32, 16: 301.55, 17: 266.71, 18: 243.99,
                    21: 309.04, 22: 321.33, 23: 283.47, 24: 278.47, 25: 309.02, 26: 294.92, 27: 252.30, 28: 235.52,
                    31: 309.74, 32: 317.15, 33: 311.47, 34: 279.64, 35: 333.06, 36: 296.73, 37: 249.48, 38: 258.87,
                    41: 258.84, 42: 256.85, 43: 283.65, 44: 286.45, 45: 265.68, 46: 306.09, 47: 263.49, 48: 268.13,
                    51: 310.47, 52: 317.14, 53: 311.81, 54: 256.70, 55: 275.80, 56: 257.35, 57: 287.21, 58: 263.44,
                    61: 340.50, 62:      0, 63: 307.23, 64: 306.92, 65: 283.94, 66: 264.04, 67: 254.19, 68:      0,
                    71: 290.54, 72: 295.58, 73: 313.08, 74: 259.79, 75: 258.95, 76: 292.69, 77: 275.09, 78: 242.91,
                    81: 321.04, 82: 286.65, 83: 261.49, 84:      0, 85: 243.37, 86: 259.59, 87: 210.20, 88: 223.17,
                    111: 343.72, 112: 333.08, 113: 386.80, 114: 365.00, 115: 299.01, 116: 0, 117: 336.51,118: 306.34, 
                    121: 401.56, 122: 359.98, 123:      0, 124: 333.20, 125: 362.41, 126: 327.71, 127: 339.82, 128: 319.30,
                    131: 384.02, 132: 369.43, 133: 223.03, 134: 244.94, 135: 325.53, 136: 329.77, 137: 321.92, 138: 361.86,
                    141: 299.28, 142: 357.88, 143: 364.20, 144: 373.72, 145: 297.64, 146: 328.10, 147: 325.22, 148: 333.75,
                    151: 377.74, 152: 380.78, 153: 367.28, 154: 336.62, 155: 321.13, 156: 355.83, 157: 318.76, 158: 335.56,
                    161: 387.74, 162: 377.52, 163: 353.05, 164: 402.35, 165: 360.38, 166: 312.83, 167: 322.64, 168: 345.87,
                    171: 316.68, 172: 347.63, 173: 374.55, 174: 354.72, 175: 323.09, 176: 345.08, 177: 337.14, 178: 303.79,
                    181: 370.28, 182: 347.17, 183: 337.94, 184: 340.84, 185: 303.30, 186: 336.85, 187: 337.40, 188: 355.60}

def apply_norm_s_id_R12212(sid: int) -> float:
    if sid < 100:
        norm = 270
    else:
        norm = 330
    return norm_s_id_R12212[sid]/norm

def filter_evt_peak(df, det_plane=True):
    if det_plane:
        tofpet_id = 5
        min_ch    = 245
        max_ch    = 290
    else:
        tofpet_id = 1
        min_ch    = 390
        max_ch    = 435

        #tofpet_id = 2
        #min_ch    = 300
        #max_ch    = 350

    df         = df[df.tofpet_id == tofpet_id]
    charge_max = df.efine_norm.max()
    return (charge_max > min_ch) & (charge_max < max_ch)


def select_evt_peak(df, det_plane=True):
    df_filter = df.groupby(['evt_number', 'cluster']).filter(filter_evt_peak, dropna=True, det_plane=det_plane)
    return df_filter


for i in range(start, start+numb):
    df0 = pd.DataFrame({})
    f = f'/analysis/{run_no}/hdf5/proc/linear_interp/files/run_{run_no}_{i:04}_trigger1_waveforms.h5'
    try:
        store = pd.HDFStore(f, 'r')
    except OSError:
        print(f'Error with file {f}')
        continue
    for key in store.keys()[i_key:i_key+n_key]:
        df = store.get(key)

        df = df.drop(columns=['ct_data', 'ctdaq', 'tac_id', 'ecoarse', 'tfine', 'tcoarse_extended', 'tfine_corrected']) #Remove unused columns
        df = df[df.cluster != -1] ## Filtering events with only one sensor

        df_coinc  = prf.compute_coincidences(df, evt_groupby=['evt_number', 'cluster'])
        max_sns_all0 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_max_sns_per_plane_new, variable='efine_corrected', det_plane=True)
        max_sns_all2 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_max_sns_per_plane_new, variable='efine_corrected', det_plane=False)
        df_coinc['max_sns0'] = max_sns_all0[df_coinc.index].values
        df_coinc['max_sns2'] = max_sns_all2[df_coinc.index].values

        df_coinc['efine_norm'] = df_coinc['efine_corrected']/df_coinc['sensor_id'].apply(apply_norm_s_id_R12252)

        tmin_all0 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_tmin_per_plane, det_plane=True)
        tmin_all2 = df_coinc.groupby(['evt_number', 'cluster']).apply(compute_tmin_per_plane, det_plane=False)
        df_coinc['tmin0'] = tmin_all0[df_coinc.index].values
        df_coinc['tmin2'] = tmin_all2[df_coinc.index].values
        
        df_peak0 = select_evt_peak(df_coinc, det_plane=True)
        df_peak2 = select_evt_peak(df_coinc, det_plane=False)

        df0 = pd.concat([df0, df_peak0], ignore_index=False, sort=False)
        df0 = pd.concat([df0, df_peak2], ignore_index=False, sort=False)

    out_file  = f'{out_path}/data_coinc_runII_ch_max_sns_norm_peak_tmin_R{run_no}_{i}_{i_key}.h5'

    df    = df0.reset_index()
    store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

print(datetime.datetime.now())
