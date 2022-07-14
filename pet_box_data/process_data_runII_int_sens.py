import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

from antea.reco import petit_reco_functions as prf

import data_taking_petalo_functions as pf


print(datetime.datetime.now())

arguments = pf.parse_args(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
run_no    = arguments.run_no
out_path  = arguments.out_path

int_area = [22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77,
            33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66, 44, 45, 54, 55]

def is_max_charge_in_int(df, det_plane, coinc_plane_4tiles, variable):
    tofpet_id, central_sns, _, _ = prf.sensor_params(det_plane, coinc_plane_4tiles)

    df = df[df.tofpet_id == tofpet_id]
    if len(df)==0:
        return False
    argmax = df[variable].argmax()

    if det_plane:
        return df.iloc[argmax].sensor_id in int_area
    else:
        return df.iloc[argmax].sensor_id in np.array(int_area) + 100


def select_evts_with_max_charge_in_int(df, evt_groupby, det_plane, coinc_plane_4tiles, variable):
    df_filter_center = df.groupby(evt_groupby).filter(is_max_charge_in_int,
                                                      dropna             = True,
                                                      det_plane          = det_plane,
                                                      coinc_plane_4tiles = coinc_plane_4tiles,
                                                      variable           = variable)
    return df_filter_center


evt_groupby        = ['evt_number', 'cluster']
variable           = 'efine_corrected'
coinc_plane_4tiles = True

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
        df = df[df.cluster != -1] ## Filtering events with only one sensor

        df_coinc  = prf.compute_coincidences(df, evt_groupby=evt_groupby)

        ## Coincidences + events not in the border
        df_int = select_evts_with_max_charge_in_int(df_coinc,
                                                    evt_groupby        = evt_groupby,
                                                    det_plane          = True,
                                                    variable           = variable,
                                                    coinc_plane_4tiles = coinc_plane_4tiles)

        df_int_c = select_evts_with_max_charge_in_int(df_int,
                                                      evt_groupby        = evt_groupby,
                                                      det_plane          = False,
                                                      variable           = variable,
                                                      coinc_plane_4tiles = coinc_plane_4tiles)

        df0 = pd.concat([df0, df_int_c], ignore_index=False, sort=False)

    out_file  = f'{out_path}/data_coinc_runII_int_sns_ch_R{run_no}_{i}.h5'

    df    = df0.reset_index()
    store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

print(datetime.datetime.now())
