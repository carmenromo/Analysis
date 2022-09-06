import os, sys

import pandas as pd
import numpy as np

import antea.database.load_db as db
from   antea.utils.map_functions import load_map

import reco.data_functions as dtf

"""
Basic analysis of run 2022
"""


file_start = int(sys.argv[1])
n_files    = int(sys.argv[2])
run        = int(sys.argv[3])
tp0        = int(sys.argv[4])
tp2        = int(sys.argv[5])


DataSiPM_pb     = db.DataSiPM('petalo', run, 'PB')
DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

folder   = f'/analysis/{run}/hdf5/proc/linear_interp/files/'
file     = 'run_{0}_{1:04d}_trigger1_waveforms.h5'
out_file = f'/home/paolafer/data/analysis/run2022/{run}/channel_analysis_he_{run}_{file_start}_{n_files}.h5'

nsipms0, nsipms2         = [], []
sensor_id0, sensor_id2   = [], []
max_charge0, max_charge2 = [], []
sum_charge0, sum_charge2 = [], []
tmin0, tmin2             = [], []
evt_number, cluster      = [], []

for i in range(file_start, file_start+n_files):
    filein = folder + file.format(run, i)
    try:
        store = pd.HDFStore(filein, 'r')
    except:
        print(f'File {filein} not found')
        continue

    print(f'Analyzing file {filein}, with {len(store.keys())} tables.')

    for key in store.keys():
        data = store.get(key)

        df       = data[data.cluster != -1]
        df       = df[df.intg_w > 0]

        df_coinc = dtf.select_coincidences(df)

        nsipms = df_coinc.groupby(['evt_number', 'cluster', 'tofpet_id'])['sensor_id'].nunique()
        n0     = nsipms[:, :, tp0].rename('nsipms0')
        n2     = nsipms[:, :, tp2].rename('nsipms2')
        nsipms0.extend(n0.values)
        nsipms2.extend(n2.values)


        boh = df_coinc.reset_index()
        max_ch  = boh.loc[boh.groupby(['evt_number', 'cluster', 'tofpet_id']).efine_corrected.idxmax()]
        max_ch0 = max_ch[max_ch.tofpet_id == tp0]
        max_ch2 = max_ch[max_ch.tofpet_id == tp2]
        sid_0   = max_ch0.sensor_id.values
        sid_2   = max_ch2.sensor_id.values
        ch_0    = max_ch0.efine_corrected.values
        ch_2    = max_ch2.efine_corrected.values
        sensor_id0.extend(sid_0)
        sensor_id2.extend(sid_2)
        max_charge0.extend(ch_0)
        max_charge2.extend(ch_2)


        tch = df_coinc.groupby(['evt_number', 'cluster', 'tofpet_id']).efine_corrected.sum()
        tch0 = tch[:, :, tp0].rename('sum_charge0')
        tch2 = tch[:, :, tp2].rename('sum_charge2')
        sum_charge0.extend(tch0.values)
        sum_charge2.extend(tch2.values)


        time = df_coinc.groupby(['evt_number', 'cluster', 'tofpet_id']).t.min()
        time0 = time[:, :, tp0].rename('tmin0')
        time2 = time[:, :, tp2].rename('tmin2')
        tmin0.extend(time0.values)
        tmin2.extend(time2.values)


    store.close()

df = pd.DataFrame({'max_charge0': max_charge0, 'max_charge2': max_charge2,
                   'sns_id0': sensor_id0, 'sns_id2': sensor_id2,
                   'sum_charge0': sum_charge0, 'sum_charge2': sum_charge2,
                   'tmin0': tmin0, 'tmin2': tmin2,
                   'nsipms0': nsipms0, 'nsipms2': nsipms2})
df = df[(df.max_charge0 > 100) | (df.max_charge2 > 100)]

store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
store.put('analysis', df, format='table', data_columns=True)
store.close()

