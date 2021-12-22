import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import antea.io  .mc_io            as mcio

""" To run this script
python get_sensors_all_info.py 0 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/tile5_centered/data_reco_info
"""

print(datetime.datetime.now())

arguments     = pbf.parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
thr_ch_start  = arguments.thr_ch_start
thr_ch_nsteps = arguments.thr_ch_nsteps
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path


corona = [11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
          51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88]


def compute_coincidences(df):
    # The dataframe must be grouped by event_id
    sensors_h = df[df.sensor_id.unique()<100].sensor_id.nunique() # Ham
    sensors_f = df[df.sensor_id.unique()>100].sensor_id.nunique() # FBK
    return sensors_h>0 and sensors_f>0

def filter_coincidences(df):
    df_filter = df.groupby('event_id').filter(compute_coincidences)
    return df_filter

# def filter_evt_with_max_charge_at_center_coinc_plane(df):
#     df = df[df.sensor_id>100]
#     if len(df)==0:
#         return False
#     argmax = df['charge'].argmax()
#     return df.iloc[argmax].sensor_id in [122, 123, 132, 133]
#
# def select_evts_with_max_charge_at_center_coinc_plane(df):
#     df_filter_center = df.filter(filter_evt_with_max_charge_at_center_coinc_plane)
#     return df_filter_center

def filter_evt_with_max_charge_at_center(df, variable):
    df = df[df.sensor_id<100]
    if len(df)==0:
        return False
    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id in [44, 45, 54, 55]

def select_evts_with_max_charge_at_center(df, variable):
    df_filter_center = df.groupby(['event_id']).filter(filter_evt_with_max_charge_at_center,
                                                       dropna=True,
                                                       variable=variable)
    return df_filter_center

def get_perc_ch_corona(df, variable='charge'):
    df     = df[df.sensor_id<100]
    tot_ch = df.groupby('event_id')[variable].sum()
    cor_ch = df[df.sensor_id.isin(corona)].groupby('event_id')[variable].sum()
    return (cor_ch/tot_ch).fillna(0)*100


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

    df_sns_resp = pd.concat([df_sns_resp, sns_response0], ignore_index=False, sort=False)

df_sns_resp_th2 = rf.find_SiPMs_over_threshold(df_sns_resp, thr)

## Coincidences:
df_coinc = filter_coincidences(df_sns_resp_th2)

## Coincidences + Centered events Hamamatsu
df_center = select_evts_with_max_charge_at_center(df_coinc, variable='charge')

perc_ch_corona  = get_perc_ch_corona(df_center, variable='charge')

df_center['perc_cor'] = perc_ch_corona[df_center.event_id].values
df_center = df_center.astype({'event_id':  'int32',
                              'sensor_id': 'int32',
                              'charge':    'int32',
                              'perc_cor':  'float64'})

store = pd.HDFStore(evt_file, "w", complib=str("zlib"), complevel=4)
store.put('data', df_center, format='table', data_columns=True)
store.close()


print(datetime.datetime.now())
