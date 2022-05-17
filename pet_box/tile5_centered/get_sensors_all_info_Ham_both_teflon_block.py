import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.reco.reco_functions       as rf
import antea.reco.petit_reco_functions as prf
import antea.io  .mc_io                as mcio

""" To run this script
python get_sensors_all_info_ANTEA_update.py 0 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
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
evt_file = f'{out_path}/get_sns_info_cov_corona_teflon_block_thr{thr}_{start}_{numb}.h5'

int_area = [22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77,
            33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66, 44, 45, 54, 55]

def is_max_charge_at_center(df, det_plane, coinc_plane_4tiles,variable):
    tofpet_id, central_sns, _, _ = prf.sensor_params(det_plane, coinc_plane_4tiles)

    df = df[df.tofpet_id == tofpet_id]
    if len(df)==0:
        return False
    argmax = df[variable].argmax()

    if det_plane:
        return df.iloc[argmax].sensor_id in int_area
    else:
        return df.iloc[argmax].sensor_id in np.array(int_area) + 100


def select_evts_with_max_charge_at_center(df, evt_groupby, det_plane, coinc_plane_4tiles, variable):
    df_filter_center = df.groupby(evt_groupby).filter(is_max_charge_at_center,
                                                      dropna             = True,
                                                      det_plane          = det_plane,
                                                      coinc_plane_4tiles = coinc_plane_4tiles,
                                                      variable           = variable)
    return df_filter_center

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

evt_groupby = ['event_id']
variable    = 'charge'
tot_mode    = False

## Coincidences:
df_coinc = prf.compute_coincidences(df_sns_resp_th2, evt_groupby)

## Coincidences + events not in the border
df_int = select_evts_with_max_charge_at_center(df_coinc,
                                                evt_groupby        = evt_groupby,
                                                det_plane          = True,
                                                variable           = variable,
                                                coinc_plane_4tiles = coinc_plane_4tiles)

df_int_c = select_evts_with_max_charge_at_center(df_int,
                                                evt_groupby        = evt_groupby,
                                                det_plane          = False,
                                                variable           = variable,
                                                coinc_plane_4tiles = coinc_plane_4tiles)

ratios = prf.compute_charge_ratio_in_corona(df_int_c,
                                            evt_groupby        = evt_groupby,
                                            variable           = variable,
                                            coinc_plane_4tiles = coinc_plane_4tiles)

df_int_c['ratio_cor'] = ratios[df_int_c.index].values
df_int_c = df_int_c.reset_index()

df_int_c = df_int_c.astype({'event_id':  'int32',
                            'sensor_id': 'int32',
                            'charge':    'int32',
                            'tofpet_id': 'int32',
                            'ratio_cor': 'float64'})

store = pd.HDFStore(evt_file, "w", complib=str("zlib"), complevel=4)
store.put('data', df_int_c, format='table', data_columns=True)
store.close()

print(datetime.datetime.now())
