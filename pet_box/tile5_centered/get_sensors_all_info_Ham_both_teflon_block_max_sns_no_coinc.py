import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.database.load_db              as db
import antea.reco    .reco_functions       as rf
import antea.reco    .petit_reco_functions as prf
import antea.io      .mc_io                as mcio
import antea.mcsim   .sensor_functions     as snsf


""" To run this script
python get_sensors_all_info_Ham_both_teflon_block_max_sns.py 0 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/tile5_centered/data_reco_info True
"""

print(datetime.datetime.now())

arguments          = pbf.parse_args_no_ths_coinc_pl_4tiles(sys.argv)
start              = arguments.first_file
numb               = arguments.n_files
in_path            = arguments.in_path
file_name          = arguments.file_name
out_path           = arguments.out_path
coinc_plane_4tiles = arguments.coinc_plane_4tiles

thr = 2
evt_file = f'{out_path}/get_sns_info_coinc_max_sns_teflon_block_NO_coinc_fluct_thr{thr}_{start}_{numb}.h5'

DataSiPM_pb     = db.DataSiPM('petalo', 11400, 'PB')
DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

def compute_max_sns_per_plane(df, variable='charge'):
    # if det_plane:
    #     df = df[df.tofpet_id == 0]
    # else:
    #     df = df[df.tofpet_id == 2]
    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id

df_sns_resp = pd.DataFrame({})
for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename = in_path + f'{file_name}.{number_str}.pet.h5'
    try:
        sns_response0 = mcio.load_mcsns_response(filename)
    except OSError:
        print(f'File {filename} does not exist')
        continue
    except KeyError:
        print(f'File {filename} is corrupt')
        continue

    df_sns_resp = pd.concat([df_sns_resp, sns_response0], ignore_index=False, sort=False)

fluct_sns_response = snsf.apply_charge_fluctuation(df_sns_resp, DataSiPM_pb_idx)
df_sns_resp_th2    = rf  .find_SiPMs_over_threshold(fluct_sns_response, thr)

df_sns_resp_th2['tofpet_id'] = df_sns_resp_th2['sensor_id'].apply(prf.tofpetid)

evt_groupby = ['event_id', 'tofpet_id']
variable    = 'charge'
tot_mode    = False

## Coincidences:
#df_coinc = prf.compute_coincidences(df_sns_resp_th2, evt_groupby)

df_coinc = df_sns_resp_th2
## Coincidences + max sns
max_sns_all = df_coinc.groupby(evt_groupby).apply(compute_max_sns_per_plane, variable='charge')
#max_sns_all2 = df_coinc.groupby(evt_groupby).apply(compute_max_sns_per_plane, variable='charge', det_plane=False)
df_coinc['max_sns'] = max_sns_all[df_coinc.index].values
#df_coinc['max_sns2'] = max_sns_all2[df_coinc.index].values

df_coinc = df_coinc.reset_index()

df_coinc = df_coinc.astype({'event_id':  'int32',
                            'sensor_id': 'int32',
                            'charge':    'int32',
                            'tofpet_id': 'int32',
                            'max_sns0': 'int32',
                            'max_sns2': 'int32'})

store = pd.HDFStore(evt_file, "w", complib=str("zlib"), complevel=4)
store.put('data', df_coinc, format='table', data_columns=True)


store.close()

print(datetime.datetime.now())
