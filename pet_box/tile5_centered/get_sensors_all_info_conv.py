import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

#import antea.reco.reco_functions   as rf
#import antea.io  .mc_io            as mcio


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
    tot_charge_evt  = df.groupby('event_id').intg_w.sum()
    max_charge_evt  = df.groupby('event_id').intg_w.max()
    touched_sns_evt = df.groupby('event_id').sensor_id.nunique()
    return tot_charge_evt, max_charge_evt, touched_sns_evt

def get_sns_info_ToT(df):
    tot_charge_evt  = df.groupby('event_id').ToT_pe.sum()
    max_charge_evt  = df.groupby('event_id').ToT_pe.max()
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
    argmax = df['intg_w'].argmax()
    return df.iloc[argmax].sensor_id in [44, 45, 54, 55]

def select_evts_with_max_charge_at_center(df):
    df_filter_center = df.filter(filter_evt_with_max_charge_at_center)
    return df_filter_center

def filter_evt_ch_corona(df):
    sens_unique = df.sensor_id.unique()
    if len(np.intersect1d(df.sensor_id.unique(), np.array(corona))):
        return True
    else:
        return False

def select_evts_ch_corona(df):
    df_filter = df.filter(filter_evt_ch_corona)
    return df_filter

def get_perc_ch_corona(df, variable='ToT_pe'):
    tot_ch = df.groupby('event_id')[variable].sum()
    cor_ch = df[df.sensor_id.isin(corona)].groupby('event_id')[variable].sum()
    return (cor_ch/tot_ch)*100

def filter_evt_percent_ch_corona(df, variable='ToT_pe', lo_p=0, hi_p=100):
    perc_cor_total = get_perc_ch_corona(df, variable=variable)
    return (perc_cor_total > lo_p) & (perc_cor_total < hi_p)

def select_evt_percent_ch_corona(df, variable='ToT_pe', lo_p=0, hi_p=100):
    df_filter = df.groupby('event_id').filter(filter_evt_percent_ch_corona,
                                                             dropna=True,
                                                             variable=variable,
                                                             lo_p=lo_p,
                                                             hi_p=hi_p)
    return df_filter


#thr = 2
evt_file   = f'{out_path}/get_sns_info_conv_{start}_{numb}'


df_sns_resp = pd.DataFrame({})
for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    filename = in_path + f'{file_name}.{number}.h5'
    try:
        #sns_response0 = mcio.load_mcsns_response(filename)
        sns_response0 = pd.read_hdf(filename, '/conv')
        sns_response0 = sns_response0[sns_response0.charge != 0]
    except OSError:
        print(f'File {filename} does not exist')
        continue

    df_sns_resp = pd.concat([df_sns_resp, sns_response0], ignore_index=False, sort=False)

#df_sns_resp_th2 = rf.find_SiPMs_over_threshold(df_sns_resp, thr)

df_h, _ = divide_sns_planes(df_sns_resp)
tot_charge_evt_h, max_charge_evt_h, touched_sns_evt_h = get_sns_info(df_h)


## Coincidences:
df_coinc      = filter_coincidences(df_sns_resp.groupby('event_id'))
df_coinc_h, _ = divide_sns_planes(df_coinc)
tot_charge_evt_coinc_h, max_charge_evt_coinc_h, touched_sns_evt_coinc_h = get_sns_info(df_coinc_h)

## Centered events:
# Hamamatsu
df_sns_resp_cent = select_evts_with_max_charge_at_center(df_sns_resp.groupby('event_id'))
df_cent_h, _     = divide_sns_planes(df_sns_resp_cent)
tot_charge_evt_cent_h, max_charge_evt_cent_h, touched_sns_evt_cent_h = get_sns_info(df_cent_h)

## Coincidences + Centered events
# Hamamatsu
df_sns_resp_coinc_cent = select_evts_with_max_charge_at_center(df_coinc.groupby('event_id'))
df_coinc_cent_h, _     = divide_sns_planes(df_sns_resp_coinc_cent)
tot_charge_evt_coinc_cent_h, max_charge_evt_coinc_cent_h, touched_sns_evt_coinc_cent_h = get_sns_info(df_coinc_cent_h)


def ToT2pe(x, df_ToT2pe):
    #The table is in ns, while the intg_w that comes from the data is watch cycles (cycle = 5ns)
    if (x*5 < np.min(df_ToT2pe['ToT_ns'])):
        return 0
    else:
        return df_ToT2pe.iloc[(np.abs(df_ToT2pe['ToT_ns']-x*5)).argmin()]['pe']

path_ToT = '/home/vherrero/CALIBRATION_FILES/'
#path_ToT = '~/Desktop/'
file_ToT = 'ToT_PE_conversion12d_high.h5'
table_ToT = pd.read_hdf(path_ToT + file_ToT, '/ToT_T2')

df_coinc_cent_h['ToT_pe'] = df_coinc_cent_h['intg_w'].apply(ToT2pe, df_ToT2pe=table_ToT)

tot_ToT_evt_coinc_cent_h, max_ToT_evt_coinc_cent_h, _ = get_sns_info_ToT(df_coinc_cent_h)

## Covered events:
df_cov = select_evts_ch_corona(df_coinc_cent_h.groupby('event_id'))
tot_charge_evt_cov_h, max_charge_evt_cov_h, touched_sns_evt_cov_h = get_sns_info(df_cov)


perc_5_tot_pe_ch,     perc_5_tot_pe_max,     perc_5_tot_pe_sns      = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=0,  hi_p=5))
print(perc_5_tot_pe_ch)
perc_5_10_tot_pe_ch,  perc_5_10_tot_pe_max,  perc_5_10_tot_pe_sns   = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=5,  hi_p=10))
perc_10_15_tot_pe_ch, perc_10_15_tot_pe_max, perc_10_15_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=10, hi_p=15))
perc_15_20_tot_pe_ch, perc_15_20_tot_pe_max, perc_15_20_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=15, hi_p=20))
perc_20_25_tot_pe_ch, perc_20_25_tot_pe_max, perc_20_25_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=20, hi_p=25))
perc_25_30_tot_pe_ch, perc_25_30_tot_pe_max, perc_25_30_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=25, hi_p=30))
perc_30_35_tot_pe_ch, perc_30_35_tot_pe_max, perc_30_35_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=30, hi_p=35))
perc_35_40_tot_pe_ch, perc_35_40_tot_pe_max, perc_35_40_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=35, hi_p=40))
perc_40_45_tot_pe_ch, perc_40_45_tot_pe_max, perc_40_45_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=40, hi_p=45))
perc_45_50_tot_pe_ch, perc_45_50_tot_pe_max, perc_45_50_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=45, hi_p=50))
perc_50_55_tot_pe_ch, perc_50_55_tot_pe_max, perc_50_55_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=50, hi_p=55))
perc_55_60_tot_pe_ch, perc_55_60_tot_pe_max, perc_55_60_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=55, hi_p=60))
perc_60_65_tot_pe_ch, perc_60_65_tot_pe_max, perc_60_65_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=60, hi_p=65))
perc_65_70_tot_pe_ch, perc_65_70_tot_pe_max, perc_65_70_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=65, hi_p=70))
perc_70_75_tot_pe_ch, perc_70_75_tot_pe_max, perc_70_75_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=70, hi_p=75))
perc_75_80_tot_pe_ch, perc_75_80_tot_pe_max, perc_75_80_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=75, hi_p=80))
perc_80_85_tot_pe_ch, perc_80_85_tot_pe_max, perc_80_85_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=80, hi_p=85))
perc_85_90_tot_pe_ch, perc_85_90_tot_pe_max, perc_85_90_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=85, hi_p=90))
perc_90_95_tot_pe_ch, perc_90_95_tot_pe_max, perc_90_95_tot_pe_sns  = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=90, hi_p=95))
perc_95_100_tot_pe_ch, perc_95_100_tot_pe_max, perc_95_100_tot_pe_sns = get_sns_info_ToT(select_evt_percent_ch_corona(df_cov, variable='ToT_pe', lo_p=95, hi_p=100))


perc_5_intgw_ch,     perc_5_intgw_max,     perc_5_intgw_sns      = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=0,  hi_p=5))
perc_5_10_intgw_ch,  perc_5_10_intgw_max,  perc_5_10_intgw_sns   = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=5,  hi_p=10))
perc_10_15_intgw_ch, perc_10_15_intgw_max, perc_10_15_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=10, hi_p=15))
perc_15_20_intgw_ch, perc_15_20_intgw_max, perc_15_20_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=15, hi_p=20))
perc_20_25_intgw_ch, perc_20_25_intgw_max, perc_20_25_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=20, hi_p=25))
perc_25_30_intgw_ch, perc_25_30_intgw_max, perc_25_30_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=25, hi_p=30))
perc_30_35_intgw_ch, perc_30_35_intgw_max, perc_30_35_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=30, hi_p=35))
perc_35_40_intgw_ch, perc_35_40_intgw_max, perc_35_40_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=35, hi_p=40))
perc_40_45_intgw_ch, perc_40_45_intgw_max, perc_40_45_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=40, hi_p=45))
perc_45_50_intgw_ch, perc_45_50_intgw_max, perc_45_50_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=45, hi_p=50))
perc_50_55_intgw_ch, perc_50_55_intgw_max, perc_50_55_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=50, hi_p=55))
perc_55_60_intgw_ch, perc_55_60_intgw_max, perc_55_60_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=55, hi_p=60))
perc_60_65_intgw_ch, perc_60_65_intgw_max, perc_60_65_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=60, hi_p=65))
perc_65_70_intgw_ch, perc_65_70_intgw_max, perc_65_70_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=65, hi_p=70))
perc_70_75_intgw_ch, perc_70_75_intgw_max, perc_70_75_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=70, hi_p=75))
perc_75_80_intgw_ch, perc_75_80_intgw_max, perc_75_80_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=75, hi_p=80))
perc_80_85_intgw_ch, perc_80_85_intgw_max, perc_80_85_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=80, hi_p=85))
perc_85_90_intgw_ch, perc_85_90_intgw_max, perc_85_90_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=85, hi_p=90))
perc_90_95_intgw_ch, perc_90_95_intgw_max, perc_90_95_intgw_sns  = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=90, hi_p=95))
perc_95_100_intgw_ch, perc_95_100_intgw_max, perc_95_100_intgw_sns = get_sns_info(select_evt_percent_ch_corona(df_cov, variable='intg_w', lo_p=95, hi_p=100))

np.savez(evt_file,  tot_charge_evt_h=tot_charge_evt_h,
        max_charge_evt_h=max_charge_evt_h,
        touched_sns_evt_h=touched_sns_evt_h,
        tot_charge_evt_coinc_h=tot_charge_evt_coinc_h,
        max_charge_evt_coinc_h=max_charge_evt_coinc_h,
        touched_sns_evt_coinc_h=touched_sns_evt_coinc_h,
        tot_charge_evt_cent_h=tot_charge_evt_cent_h,
        max_charge_evt_cent_h=max_charge_evt_cent_h,
        touched_sns_evt_cent_h=touched_sns_evt_cent_h,
        tot_charge_evt_coinc_cent_h=tot_charge_evt_coinc_cent_h,
        max_charge_evt_coinc_cent_h=max_charge_evt_coinc_cent_h,
        touched_sns_evt_coinc_cent_h=touched_sns_evt_coinc_cent_h,
        tot_ToT_evt_coinc_cent_h=tot_ToT_evt_coinc_cent_h,
        max_ToT_evt_coinc_cent_h=max_ToT_evt_coinc_cent_h,
        tot_charge_evt_cov_h=tot_charge_evt_cov_h,
        max_charge_evt_cov_h=max_charge_evt_cov_h,
        perc_5_tot_pe_ch=perc_5_tot_pe_ch,
        perc_5_tot_pe_max=perc_5_tot_pe_max,
        perc_5_tot_pe_sns=perc_5_tot_pe_sns,
        perc_5_10_tot_pe_ch=perc_5_10_tot_pe_ch,
        perc_5_10_tot_pe_max=perc_5_10_tot_pe_max,
        perc_5_10_tot_pe_sns=perc_5_10_tot_pe_sns,
        perc_10_15_tot_pe_ch=perc_10_15_tot_pe_ch,
        perc_10_15_tot_pe_max=perc_10_15_tot_pe_max,
        perc_10_15_tot_pe_sns=perc_10_15_tot_pe_sns,
        perc_15_20_tot_pe_ch=perc_15_20_tot_pe_ch,
        perc_15_20_tot_pe_max=perc_15_20_tot_pe_max,
        perc_15_20_tot_pe_sns=perc_15_20_tot_pe_sns,
        perc_20_25_tot_pe_ch=perc_20_25_tot_pe_ch,
        perc_20_25_tot_pe_max=perc_20_25_tot_pe_max,
        perc_20_25_tot_pe_sns=perc_20_25_tot_pe_sns,
        perc_25_30_tot_pe_ch=perc_25_30_tot_pe_ch,
        perc_25_30_tot_pe_max=perc_25_30_tot_pe_max,
        perc_25_30_tot_pe_sns=perc_25_30_tot_pe_sns,
        perc_30_35_tot_pe_ch=perc_30_35_tot_pe_ch,
        perc_30_35_tot_pe_max=perc_30_35_tot_pe_max,
        perc_30_35_tot_pe_sns=perc_30_35_tot_pe_sns,
        perc_35_40_tot_pe_ch=perc_35_40_tot_pe_ch,
        perc_35_40_tot_pe_max=perc_35_40_tot_pe_max,
        perc_35_40_tot_pe_sns=perc_35_40_tot_pe_sns,
        perc_40_45_tot_pe_ch=perc_40_45_tot_pe_ch,
        perc_40_45_tot_pe_max=perc_40_45_tot_pe_max,
        perc_40_45_tot_pe_sns=perc_40_45_tot_pe_sns,
        perc_45_50_tot_pe_ch=perc_45_50_tot_pe_ch,
        perc_45_50_tot_pe_max=perc_45_50_tot_pe_max,
        perc_45_50_tot_pe_sns=perc_45_50_tot_pe_sns,
        perc_50_55_tot_pe_ch=perc_50_55_tot_pe_ch,
        perc_50_55_tot_pe_max=perc_50_55_tot_pe_max,
        perc_50_55_tot_pe_sns=perc_50_55_tot_pe_sns,
        perc_55_60_tot_pe_ch=perc_55_60_tot_pe_ch,
        perc_55_60_tot_pe_max=perc_55_60_tot_pe_max,
        perc_55_60_tot_pe_sns=perc_55_60_tot_pe_sns,
        perc_60_65_tot_pe_ch=perc_60_65_tot_pe_ch,
        perc_60_65_tot_pe_max=perc_60_65_tot_pe_max,
        perc_60_65_tot_pe_sns=perc_60_65_tot_pe_sns,
        perc_65_70_tot_pe_ch=perc_65_70_tot_pe_ch,
        perc_65_70_tot_pe_max=perc_65_70_tot_pe_max,
        perc_65_70_tot_pe_sns=perc_65_70_tot_pe_sns,
        perc_70_75_tot_pe_ch=perc_70_75_tot_pe_ch,
        perc_70_75_tot_pe_max=perc_70_75_tot_pe_max,
        perc_70_75_tot_pe_sns=perc_70_75_tot_pe_sns,
        perc_75_80_tot_pe_ch=perc_75_80_tot_pe_ch,
        perc_75_80_tot_pe_max=perc_75_80_tot_pe_max,
        perc_75_80_tot_pe_sns=perc_75_80_tot_pe_sns,
        perc_80_85_tot_pe_ch=perc_80_85_tot_pe_ch,
        perc_80_85_tot_pe_max=perc_80_85_tot_pe_max,
        perc_80_85_tot_pe_sns=perc_80_85_tot_pe_sns,
        perc_85_90_tot_pe_ch=perc_85_90_tot_pe_ch,
        perc_85_90_tot_pe_max=perc_85_90_tot_pe_max,
        perc_85_90_tot_pe_sns=perc_85_90_tot_pe_sns,
        perc_90_95_tot_pe_ch=perc_90_95_tot_pe_ch,
        perc_90_95_tot_pe_max=perc_90_95_tot_pe_max,
        perc_90_95_tot_pe_sns=perc_90_95_tot_pe_sns,
        perc_95_100_tot_pe_ch=perc_95_100_tot_pe_ch,
        perc_95_100_tot_pe_max=perc_95_100_tot_pe_max,
        perc_5_intgw_ch=perc_5_intgw_ch,
        perc_5_intgw_max=perc_5_intgw_max,
        perc_5_intgw_sns=perc_5_intgw_sns,
        perc_5_10_intgw_ch=perc_5_10_intgw_ch,
        perc_5_10_intgw_max=perc_5_10_intgw_max,
        perc_5_10_intgw_sns=perc_5_10_intgw_sns,
        perc_10_15_intgw_ch=perc_10_15_intgw_ch,
        perc_10_15_intgw_max=perc_10_15_intgw_max,
        perc_10_15_intgw_sns=perc_10_15_intgw_sns,
        perc_15_20_intgw_ch=perc_15_20_intgw_ch,
        perc_15_20_intgw_max=perc_15_20_intgw_max,
        perc_15_20_intgw_sns=perc_15_20_intgw_sns,
        perc_20_25_intgw_ch=perc_20_25_intgw_ch,
        perc_20_25_intgw_max=perc_20_25_intgw_max,
        perc_20_25_intgw_sns=perc_20_25_intgw_sns,
        perc_25_30_intgw_ch=perc_25_30_intgw_ch,
        perc_25_30_intgw_max=perc_25_30_intgw_max,
        perc_25_30_intgw_sns=perc_25_30_intgw_sns,
        perc_30_35_intgw_ch=perc_30_35_intgw_ch,
        perc_30_35_intgw_max=perc_30_35_intgw_max,
        perc_30_35_intgw_sns=perc_30_35_intgw_sns,
        perc_35_40_intgw_ch=perc_35_40_intgw_ch,
        perc_35_40_intgw_max=perc_35_40_intgw_max,
        perc_35_40_intgw_sns=perc_35_40_intgw_sns,
        perc_40_45_intgw_ch=perc_40_45_intgw_ch,
        perc_40_45_intgw_max=perc_40_45_intgw_max,
        perc_40_45_intgw_sns=perc_40_45_intgw_sns,
        perc_45_50_intgw_ch=perc_45_50_intgw_ch,
        perc_45_50_intgw_max=perc_45_50_intgw_max,
        perc_45_50_intgw_sns=perc_45_50_intgw_sns,
        perc_50_55_intgw_ch=perc_50_55_intgw_ch,
        perc_50_55_intgw_max=perc_50_55_intgw_max,
        perc_50_55_intgw_sns=perc_50_55_intgw_sns,
        perc_55_60_intgw_ch=perc_55_60_intgw_ch,
        perc_55_60_intgw_max=perc_55_60_intgw_max,
        perc_55_60_intgw_sns=perc_55_60_intgw_sns,
        perc_60_65_intgw_ch=perc_60_65_intgw_ch,
        perc_60_65_intgw_max=perc_60_65_intgw_max,
        perc_60_65_intgw_sns=perc_60_65_intgw_sns,
        perc_65_70_intgw_ch=perc_65_70_intgw_ch,
        perc_65_70_intgw_max=perc_65_70_intgw_max,
        perc_65_70_intgw_sns=perc_65_70_intgw_sns,
        perc_70_75_intgw_ch=perc_70_75_intgw_ch,
        perc_70_75_intgw_max=perc_70_75_intgw_max,
        perc_70_75_intgw_sns=perc_70_75_intgw_sns,
        perc_75_80_intgw_ch=perc_75_80_intgw_ch,
        perc_75_80_intgw_max=perc_75_80_intgw_max,
        perc_75_80_intgw_sns=perc_75_80_intgw_sns,
        perc_80_85_intgw_ch=perc_80_85_intgw_ch,
        perc_80_85_intgw_max=perc_80_85_intgw_max,
        perc_80_85_intgw_sns=perc_80_85_intgw_sns,
        perc_85_90_intgw_ch=perc_85_90_intgw_ch,
        perc_85_90_intgw_max=perc_85_90_intgw_max,
        perc_85_90_intgw_sns=perc_85_90_intgw_sns,
        perc_90_95_intgw_ch=perc_90_95_intgw_ch,
        perc_90_95_intgw_max=perc_90_95_intgw_max,
        perc_90_95_intgw_sns=perc_90_95_intgw_sns,
        perc_95_100_intgw_ch=perc_95_100_intgw_ch,
        perc_95_100_intgw_max=perc_95_100_intgw_max,
        perc_95_100_intgw_sns=perc_95_100_intgw_sns)

print(datetime.datetime.now())
