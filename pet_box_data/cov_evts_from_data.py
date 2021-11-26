import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

from antea.reco import data_reco_functions as drf

import data_taking_petalo_functions as pf


print(datetime.datetime.now())

arguments = pf.parse_args(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
run_no    = arguments.run_no
out_path  = arguments.out_path


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

        df['intg_w_ToT'] = df['t2'] - df['t1']
        df = df[(df['intg_w_ToT']>0) & (df['intg_w_ToT']<500)]

        df_coinc  = drf.compute_coincidences(df)
        df_center = drf.select_evts_with_max_charge_at_center(df_coinc, tot_mode=True)
        df_center['ToT_pe'] = from_ToT_to_pes(df_center['intg_w_ToT']*5) #### This function takes the time in ns, not in cycles!!!

        df_cov = drf.select_contained_evts_in_det_plane(df_center)

        df0 = pd.concat([df0, df_cov], ignore_index=False, sort=False)

    evt_file  = f'{out_path}/data_coinc_area0_intgw_perc_ch_R{run_no}_{i}'
    pf.save_df(df0, evt_file)

print(datetime.datetime.now())
