import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import antea.reco.reco_functions       as rf
import antea.reco.petit_reco_functions as prf
import antea.io  .mc_io                as mcio

""" To run this script
python get_sensors_all_info_RunI_MC.py 0 1 /Users/carmenromoluque/Desktop/ PetBox_mix_HamVUV_FBK_tile5centered
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/tile5_centered/data_reco_info
"""

print(datetime.datetime.now())

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

thr = 2
evt_file   = f'{out_path}/get_sns_info_cov_corona_thr{thr}_{start}_{numb}.h5'

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

df_sns_resp_th2 = rf.find_SiPMs_over_threshold(df_sns_resp, thr)

df_sns_resp_th2['tofpet_id'] = df_sns_resp_th2['sensor_id'].apply(prf.tofpetid)

evt_groupby        = ['event_id']
variable           = 'charge'
tot_mode           = False
coinc_plane_4tiles = False

## Coincidences:
df_coinc = prf.compute_coincidences(df_sns_resp_th2, evt_groupby)

## Coincidences + Centered events det plane
df_center = prf.select_evts_with_max_charge_at_center(df_coinc,
                                                      evt_groupby        = evt_groupby,
                                                      det_plane          = True,
                                                      variable           = variable,
                                                      tot_mode           = tot_mode,
                                                      coinc_plane_4tiles = coinc_plane_4tiles)

ratios = prf.compute_charge_ratio_in_corona(df_center,
                                            evt_groupby        = evt_groupby,
                                            variable           = variable,
                                            coinc_plane_4tiles = coinc_plane_4tiles)
df_center['ratio_cor'] = ratios[df_center.index].values
df_center = df_center.reset_index()
df_center = df_center.astype({'event_id':  'int32',
                                  'sensor_id': 'int32',
                                  'charge':    'int32',
                                  'tofpet_id': 'int32',
                                  'ratio_cor': 'float64'})

store = pd.HDFStore(evt_file, "w", complib=str("zlib"), complevel=4)
store.put('data', df_center, format='table', data_columns=True)
store.close()

print(datetime.datetime.now())
