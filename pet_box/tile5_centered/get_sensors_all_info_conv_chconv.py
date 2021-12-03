import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import antea.database.load_db as db
from invisible_cities.reco.sensor_functions import charge_fluctuation


""" To run this script
python get_sensors_all_info_conv_chconv.py 0 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/tile5_centered/data_reco_info
"""

print(datetime.datetime.now())


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file', type = int, help = "first file (inclusive)"        )
    parser.add_argument('n_files'   , type = int, help = "number of files to analize"    )
    parser.add_argument('in_path'   ,             help = "input files path"              )
    parser.add_argument('file_name' ,             help = "name of input files"           )
    parser.add_argument('out_path'  ,             help = "output files path"             )
    #parser.add_argument('variable'  ,             help = "Variable to extract max charge")
    return parser.parse_args()

arguments = parse_args(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path


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
    df     = df[df.sensor_id<100]
    tot_ch = df.groupby('event_id')[variable].sum()
    cor_ch = df[df.sensor_id.isin(corona)].groupby('event_id')[variable].sum()
    return (cor_ch/tot_ch).fillna(0)*100


DataSiPM_pb     = db.DataSiPM('petalo', 0, 'PB')
DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

def fluctuate_charge(df_h, variable='charge', peak_pe=1600):
    xe_fluct = 0.06
    x = np.sqrt(peak_pe)*xe_fluct

    DataSiPM_pb_idx.Sigma = np.repeat(x, len(DataSiPM_pb_idx))

    ch      = df_h[variable].values
    sipmrd  = ch[:, np.newaxis]

    pe_resolution = DataSiPM_pb_idx.Sigma / DataSiPM_pb_idx.adc_to_pes
    touched_sipms = df_h.sensor_id.values
    pe_res        = pe_resolution[touched_sipms]

    sipm_fluct = np.array(tuple(map(charge_fluctuation, sipmrd, pe_res)))
    fluct_charge = sipm_fluct.flatten()
    return fluct_charge


def fluctuate_sum_charge_det_plane(df, variable='charge_conv', peak_pe=1600):
    xe_fluct           = 0.06
    sum_charges        = df.groupby('event_id')[variable].sum()
    pe_res             = np.sqrt(peak_pe)*xe_fluct
    sum_ch_fluct       = charge_fluctuation(sum_charges, pe_res)
    return sum_ch_fluct


evt_file   = f'{out_path}/get_sns_info_conv_chconv_{start}_{numb}'
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
df_sns_resp_coinc_cent = select_evts_with_max_charge_at_center(df_coinc, variable='ToT')
perc_ch_corona = get_perc_ch_corona(df_sns_resp_coinc_cent, variable='charge_conv')
df_sns_resp_coinc_cent = df_sns_resp_coinc_cent.set_index(['event_id'])

charge_conv_sum = df_sns_resp_coinc_cent.groupby('event_id').charge_conv.sum()
evt_ids         = df_sns_resp_coinc_cent.groupby('event_id').charge_conv.sum().index
df_sns_resp_coinc_cent['charge_conv_sum'] = pd.Series(data=charge_conv_sum, index=evt_ids)[df_sns_resp_coinc_cent.index].values

#perc_cor --> ratio
df_sns_resp_coinc_cent['ratio_conv']      = perc_ch_corona[df_sns_resp_coinc_cent.index].values

charge_conv_fluct1 = fluctuate_charge(df_sns_resp_coinc_cent, variable='charge_conv', peak_pe=1450)
df_sns_resp_coinc_cent['f1_conv_1450'] = charge_conv_fluct1

df_fluct2 = fluctuate_sum_charge_det_plane(df_sns_resp_coinc_cent, variable='charge_conv', peak_pe=1450)

df_fluct2['f3_conv'] = df_fluct2.charge_conv * df_fluct2.f2_conv_1450 / df_fluct2.charge_conv_sum
perc_ch_corona2 = get_perc_ch_corona(df_fluct2, variable='f3_conv')
df_fluct2['ratio_fl_conv'] = perc_ch_corona2[df_fluct2.index].values

np.savez(evt_file, event_id       =df_fluct2.index,
                   sensor_id      =df_fluct2.sensor_id,
                   charge_data    =df_fluct2.charge_data,
                   charge_conv    =df_fluct2.charge_conv,
                   charge_mc      =df_fluct2.charge_mc,
                   ToT            =df_fluct2.ToT,
                   charge_conv_sum=df_fluct2.charge_conv_sum,
                   ratio_conv     =df_fluct2.ratio_conv,
                   f1_conv_1450   =df_fluct2.f1_conv_1450,
                   f2_conv_1450   =df_fluct2.f2_conv_1450,
                   f3_conv        =df_fluct2.f3_conv,
                   ratio_fl_conv  =df_fluct2.ratio_fl_conv)
print(datetime.datetime.now())
