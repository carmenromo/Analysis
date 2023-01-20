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


def compute_no_coincidences(df, evt_groupby):
    """
    Returns the events in which only one plane have detected charge (NO coinc).
    """
    nplanes     = df.groupby(evt_groupby)['tofpet_id'].nunique()
    df_idx      = df.set_index(evt_groupby)
    df_no_coinc = df_idx.loc[nplanes[nplanes == 1].index]

    return df_no_coinc


def compute_max_sns_per_plane(df, variable='efine_corrected', det_plane=True):
    if det_plane:
        df = df[df.tofpet_id == tofpet_d]
    else:
        df = df[df.tofpet_id == tofpet_c]
    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id #channel_id


for i in range(start, start+numb):
    df0 = pd.DataFrame({})
    f = f'/analysis/{run_no}/hdf5/proc/linear_interp/files/run_{run_no}_{i:04}_trigger1_waveforms.h5'
    try:
        store = pd.HDFStore(f, 'r')
    except OSError:
        print(f'Error with file {f}')
        continue
    for key in store.keys()[:10]:
        df = store.get(key)
        df = df[df.cluster != -1] ## Filtering events with only one sensor

        df_coinc  = compute_no_coincidences(df, evt_groupby=['evt_number', 'cluster'])
        df_coinc0 = df_coinc[df_coinc.tofpet_id==tofpet_d]
        df_coinc2 = df_coinc[df_coinc.tofpet_id==tofpet_c]
        max_sns_all0 = df_coinc0.groupby(['evt_number', 'cluster']).apply(compute_max_sns_per_plane, variable='efine_corrected', det_plane=True)
        max_sns_all2 = df_coinc2.groupby(['evt_number', 'cluster']).apply(compute_max_sns_per_plane, variable='efine_corrected', det_plane=False)
        df_coinc0['max_sns0'] = max_sns_all0[df_coinc0.index].values
        df_coinc2['max_sns2'] = max_sns_all2[df_coinc2.index].values
        df0 = pd.concat([df0, df_coinc0], ignore_index=False, sort=False)
        df0 = pd.concat([df0, df_coinc2], ignore_index=False, sort=False)

    out_file  = f'{out_path}/data_NO_coinc_runII_ch_max_sns_R{run_no}_{i}.h5'

    df    = df0.reset_index()
    store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

print(datetime.datetime.now())
