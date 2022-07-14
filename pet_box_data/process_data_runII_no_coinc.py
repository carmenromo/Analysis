import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

import data_taking_petalo_functions as pf


print(datetime.datetime.now())

arguments = pf.parse_args(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
run_no    = arguments.run_no
out_path  = arguments.out_path

def compute_no_coincidences(df, evt_groupby):
    """
    Returns the events in which only one plane have detected charge (NO coinc).
    """
    nplanes     = df.groupby(evt_groupby)['tofpet_id'].nunique()
    df_idx      = df.set_index(evt_groupby)
    df_no_coinc = df_idx.loc[nplanes[nplanes == 1].index]

    return df_no_coinc


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

        df_no_coinc  = compute_no_coincidences(df, evt_groupby=['evt_number', 'cluster'])
        df0 = pd.concat([df0, df_no_coinc], ignore_index=False, sort=False)

    out_file  = f'{out_path}/data_no_coinc_runII_ch_R{run_no}_{i}.h5'

    df    = df0.reset_index()
    store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

print(datetime.datetime.now())
