import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf


""" To run this script
python get_sensors_all_info_conv.py 0 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
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
    df = df[df.sensor_id<100]
    tot_ch = df.groupby('event_id')[variable].sum()
    cor_ch = df[df.sensor_id.isin(corona)].groupby('event_id')[variable].sum()
    return (cor_ch/tot_ch)*100


evt_file   = f'{out_path}/get_sns_info_conv_{start}_{numb}'

df_sns_resp = pd.DataFrame({})
for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    filename = in_path + f'{file_name}.{number}.h5'
    try:
        sns_response0 = pd.read_hdf(filename, '/conv')
        sns_response0 = sns_response0[sns_response0.ToT>0]
    except OSError:
        print(f'File {filename} does not exist')
        continue

    df_sns_resp = pd.concat([df_sns_resp, sns_response0], ignore_index=False, sort=False)


## Coincidences:
df_coinc = filter_coincidences(df_sns_resp)

## Coincidences + Centered events Hamamatsu
df_sns_resp_coinc_cent1 = select_evts_with_max_charge_at_center(df_coinc, variable='charge_data')
df_sns_resp_coinc_cent2 = select_evts_with_max_charge_at_center(df_coinc, variable='charge_conv')
df_sns_resp_coinc_cent3 = select_evts_with_max_charge_at_center(df_coinc, variable='charge_mc')
df_sns_resp_coinc_cent4 = select_evts_with_max_charge_at_center(df_coinc, variable='ToT')

perc_ch_corona1 = get_perc_ch_corona(df_sns_resp_coinc_cent1, variable='charge_data')
perc_ch_corona2 = get_perc_ch_corona(df_sns_resp_coinc_cent2, variable='charge_conv')
perc_ch_corona3 = get_perc_ch_corona(df_sns_resp_coinc_cent3, variable='charge_mc')

df_sns_resp_coinc_cent1 = df_sns_resp_coinc_cent1.set_index(['event_id'])
df_sns_resp_coinc_cent2 = df_sns_resp_coinc_cent2.set_index(['event_id'])
df_sns_resp_coinc_cent3 = df_sns_resp_coinc_cent3.set_index(['event_id'])

df_sns_resp_coinc_cent1['perc_cor'] = perc_ch_corona1[df_sns_resp_coinc_cent1.index].values
df_sns_resp_coinc_cent2['perc_cor'] = perc_ch_corona2[df_sns_resp_coinc_cent2.index].values
df_sns_resp_coinc_cent3['perc_cor'] = perc_ch_corona3[df_sns_resp_coinc_cent3.index].values


np.savez(evt_file, event_id1   =df_sns_resp_coinc_cent1.index,
                   sensor_id1  =df_sns_resp_coinc_cent1.sensor_id,
                   charge_data1=df_sns_resp_coinc_cent1.charge_data,
                   charge_conv1=df_sns_resp_coinc_cent1.charge_conv,
                   charge_mc1  =df_sns_resp_coinc_cent1.charge_mc,
                   ToT1        =df_sns_resp_coinc_cent1.ToT,
                   perc_cor1   =df_sns_resp_coinc_cent1.perc_cor,
                   event_id2   =df_sns_resp_coinc_cent2.index,
                   sensor_id2  =df_sns_resp_coinc_cent2.sensor_id,
                   charge_data2=df_sns_resp_coinc_cent2.charge_data,
                   charge_conv2=df_sns_resp_coinc_cent2.charge_conv,
                   charge_mc2  =df_sns_resp_coinc_cent2.charge_mc,
                   ToT2        =df_sns_resp_coinc_cent2.ToT,
                   perc_cor2   =df_sns_resp_coinc_cent2.perc_cor,
                   event_id3   =df_sns_resp_coinc_cent3.index,
                   sensor_id3  =df_sns_resp_coinc_cent3.sensor_id,
                   charge_data3=df_sns_resp_coinc_cent3.charge_data,
                   charge_conv3=df_sns_resp_coinc_cent3.charge_conv,
                   charge_mc3  =df_sns_resp_coinc_cent3.charge_mc,
                   ToT3        =df_sns_resp_coinc_cent3.ToT,
                   perc_cor3   =df_sns_resp_coinc_cent3.perc_cor,
                   event_id4   =df_sns_resp_coinc_cent4.index,
                   sensor_id4  =df_sns_resp_coinc_cent4.sensor_id,
                   charge_data4=df_sns_resp_coinc_cent4.charge_data,
                   charge_conv4=df_sns_resp_coinc_cent4.charge_conv,
                   charge_mc4  =df_sns_resp_coinc_cent4.charge_mc,
                   ToT4        =df_sns_resp_coinc_cent4.ToT)

print(datetime.datetime.now())
