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

area0 = [44, 45, 54, 55]
area1 = [33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66]
area2 = [22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77]
area5 = area0 + area1 + area2

corona = [11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
          51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88]

def divide_sns_planes(df):
    df_h = df[df.sensor_id<100]
    df_f = df[df.sensor_id>100]
    return df_h, df_f

def get_sns_info(df):
    tot_charge_evt  = df.groupby('event_id').charge.sum()
    max_charge_evt  = df.groupby('event_id').charge.max()
    touched_sns_evt = df.groupby('event_id').sensor_id.nunique()
    return tot_charge_evt, max_charge_evt, touched_sns_evt

def compute_coincidences(df):
    # The dataframe must be grouped by event_id
    sensors_h = df[df.sensor_id.unique()<100].sensor_id.nunique() # Ham
    sensors_f = df[df.sensor_id.unique()>100].sensor_id.nunique() # FBK
    if sensors_h>0 and sensors_f>0: return True
    else: return False

def filter_coincidences(df):
    df_filter = df.filter(compute_coincidences)
    return df_filter

def filter_evt_with_max_charge_at_center(df):
    df = df[df.sensor_id<100]
    if len(df)==0:
        return False
    argmax = df['charge'].argmax()
    return df.iloc[argmax].sensor_id in [44, 45, 54, 55]

def select_evts_with_max_charge_at_center(df):
    df_filter_center = df.filter(filter_evt_with_max_charge_at_center)
    return df_filter_center

def filter_evt_with_max_charge_at_center_coinc_plane(df):
    df = df[df.sensor_id>100]
    if len(df)==0:
        return False
    argmax = df['charge'].argmax()
    return df.iloc[argmax].sensor_id in [122, 123, 132, 133]

def select_evts_with_max_charge_at_center_coinc_plane(df):
    df_filter_center = df.filter(filter_evt_with_max_charge_at_center_coinc_plane)
    return df_filter_center

# def filter_covered_evt(df):
#     df = df[df.sensor_id<100]
#     sens_unique = df.sensor_id.unique()
#     if len(sens_unique) > 1: return set(sens_unique).issubset(set(area5))
#     else: return False
#
# def select_covered_evts(df):
#     df_filter_center = df.filter(filter_covered_evt)
#     return df_filter_center

def get_perc_ch_corona(df):
    tot_ch = df.groupby('event_id').charge.sum()
    cor_ch = df[df.sensor_id.isin(corona)].groupby('event_id').charge.sum()
    return (cor_ch/tot_ch).fillna(0)*100

def filter_evt_percent_ch_corona(df, lo_p=0, hi_p=100):
    perc_cor_total = get_perc_ch_corona(df)
    return (perc_cor_total > lo_p) & (perc_cor_total < hi_p)

def select_evt_percent_ch_corona(df, lo_p=0, hi_p=100):
    df_filter = df.groupby('event_id').filter(filter_evt_percent_ch_corona,
                                                             dropna=True,
                                                             lo_p=lo_p,
                                                             hi_p=hi_p)
    return df_filter


thr = 2
evt_file   = f'{out_path}/get_sns_info_thr{thr}_{start}_{numb}'


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

df_h, df_f = divide_sns_planes(df_sns_resp)
tot_charge_evt_h, _, _ = get_sns_info(df_h)
#tot_charge_evt_f, max_charge_evt_f, touched_sns_evt_f = get_sns_info(df_f)

df_th2_h, _                = divide_sns_planes(df_sns_resp_th2)
tot_charge_evt_th2_h, _, _ = get_sns_info(df_th2_h)
#tot_charge_evt_th2_f, max_charge_evt_th2_f, touched_sns_evt_th2_f = get_sns_info(df_th2_f)

## Coincidences:
df_coinc      = filter_coincidences(df_sns_resp_th2.groupby('event_id'))
df_coinc_h, _ = divide_sns_planes(df_coinc)
tot_charge_evt_coinc_h, _, _ = get_sns_info(df_coinc_h)
#tot_charge_evt_coinc_f, max_charge_evt_coinc_f, touched_sns_evt_coinc_f = get_sns_info(df_coinc_f)

## Centered events:
# Hamamatsu
df_sns_resp_cent = select_evts_with_max_charge_at_center(df_sns_resp_th2.groupby('event_id'))
df_cent_h, _     = divide_sns_planes(df_sns_resp_cent)
tot_charge_evt_cent_h, _, _ = get_sns_info(df_cent_h)
# FBK
# df_sns_resp_cent_c = select_evts_with_max_charge_at_center_coinc_plane(df_sns_resp_th2.groupby('event_id'))
# _, df_cent_f       = divide_sns_planes(df_sns_resp_cent_c)
# tot_charge_evt_cent_f, max_charge_evt_cent_f, touched_sns_evt_cent_f = get_sns_info(df_cent_f)

## Coincidences + Centered events
# Hamamatsu
df_sns_resp_coinc_cent = select_evts_with_max_charge_at_center(df_coinc.groupby('event_id'))
df_coinc_cent_h, _     = divide_sns_planes(df_sns_resp_coinc_cent)
tot_charge_evt_coinc_cent_h, _, _ = get_sns_info(df_coinc_cent_h)
# FBK
# df_sns_resp_coinc_cent_c = select_evts_with_max_charge_at_center_coinc_plane(df_coinc.groupby('event_id'))
# _, df_coinc_cent_f       = divide_sns_planes(df_sns_resp_coinc_cent_c)
# tot_charge_evt_coinc_cent_f, max_charge_evt_coinc_cent_f, touched_sns_evt_coinc_cent_f = get_sns_info(df_coinc_cent_f)


## Covered events:
# df_sns_resp_cov = select_covered_evts(df_sns_resp_th2.groupby('event_id'))
# df_cov_h, _     = divide_sns_planes(df_sns_resp_cov)
# tot_charge_evt_cov_h, max_charge_evt_cov_h, touched_sns_evt_cov_h = get_sns_info(df_cov_h)

perc_ch_corona = get_perc_ch_corona(df_coinc_cent_h)

df_coinc_cent_h['perc_cor'] = perc_ch_corona[df_coinc_cent_h.event_id].values

np.savez(evt_file,  tot_charge_evt_h=tot_charge_evt_h, tot_charge_evt_th2_h=tot_charge_evt_th2_h, tot_charge_evt_coinc_h=tot_charge_evt_coinc_h,
         tot_charge_evt_cent_h=tot_charge_evt_cent_h, tot_charge_evt_coinc_cent_h=tot_charge_evt_coinc_cent_h, perc_ch_corona=perc_ch_corona.values,
         df_event_id=df_coinc_cent_h.event_id, df_sensor_id=df_coinc_cent_h.sensor_id, df_charge=df_coinc_cent_h.charge,
         df_time_bin=df_coinc_cent_h.time_bin, df_perc_cor=df_coinc_cent_h.perc_cor)

# np.savez(evt_file,  tot_charge_evt_h=tot_charge_evt_h, max_charge_evt_h=max_charge_evt_h, touched_sns_evt_h=touched_sns_evt_h,
#         tot_charge_evt_f=tot_charge_evt_f, max_charge_evt_f=max_charge_evt_f, touched_sns_evt_f=touched_sns_evt_f,
#         tot_charge_evt_th2_h=tot_charge_evt_th2_h, max_charge_evt_th2_h=max_charge_evt_th2_h, touched_sns_evt_th2_h=touched_sns_evt_th2_h,
#         tot_charge_evt_th2_f=tot_charge_evt_th2_f, max_charge_evt_th2_f=max_charge_evt_th2_f, touched_sns_evt_th2_f=touched_sns_evt_th2_f,
#         tot_charge_evt_coinc_h=tot_charge_evt_coinc_h, max_charge_evt_coinc_h=max_charge_evt_coinc_h, touched_sns_evt_coinc_h=touched_sns_evt_coinc_h,
#         tot_charge_evt_coinc_f=tot_charge_evt_coinc_f, max_charge_evt_coinc_f=max_charge_evt_coinc_f, touched_sns_evt_coinc_f=touched_sns_evt_coinc_f,
#         tot_charge_evt_cent_h=tot_charge_evt_cent_h, max_charge_evt_cent_h=max_charge_evt_cent_h, touched_sns_evt_cent_h=touched_sns_evt_cent_h,
#         tot_charge_evt_cent_f=tot_charge_evt_cent_f, max_charge_evt_cent_f=max_charge_evt_cent_f, touched_sns_evt_cent_f=touched_sns_evt_cent_f,
#         tot_charge_evt_coinc_cent_h=tot_charge_evt_coinc_cent_h, max_charge_evt_coinc_cent_h=max_charge_evt_coinc_cent_h, touched_sns_evt_coinc_cent_h=touched_sns_evt_coinc_cent_h,
#         tot_charge_evt_coinc_cent_f=tot_charge_evt_coinc_cent_f, max_charge_evt_coinc_cent_f=max_charge_evt_coinc_cent_f, touched_sns_evt_coinc_cent_f=touched_sns_evt_coinc_cent_f,
#         tot_charge_evt_cov_h=tot_charge_evt_cov_h, max_charge_evt_cov_h=max_charge_evt_cov_h, touched_sns_evt_cov_h=touched_sns_evt_cov_h)

print(datetime.datetime.now())
