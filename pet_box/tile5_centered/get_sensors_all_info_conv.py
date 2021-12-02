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

def str2bool(v):
    """
    This function is added because the argparse add_argument('use_db_gain_seeds', type=bool)
    was not working in False case, everytime True was taken.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int,    help = "first file (inclusive)"        )
    parser.add_argument('n_files'      , type = int,    help = "number of files to analize"    )
    parser.add_argument('in_path'      ,                help = "input files path"              )
    parser.add_argument('file_name'    ,                help = "name of input files"           )
    parser.add_argument('out_path'     ,                help = "output files path"             )
    parser.add_argument('variable'     ,                help = "Variable to extract max charge")
    parser.add_argument('apply_thr'    , type=str2bool, help = "Apply threshold in charge"     )
    return parser.parse_args()

arguments = parse_args(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
in_path   = arguments.in_path
file_name = arguments.file_name
out_path  = arguments.out_path
variable  = arguments.variable
apply_thr = arguments.apply_thr


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
    return (cor_ch/tot_ch).fillna(0)*100

thr = 9.9 #pes
evt_file   = f'{out_path}/get_sns_info_conv_{variable}_{apply_thr}_thr{thr}_{start}_{numb}'
df_sns_resp = pd.DataFrame({})
for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    filename = in_path + f'{file_name}.{number}.h5'
    try:
        sns_response0 = pd.read_hdf(filename, '/conv')
        if apply_thr:
            sns_response0 = sns_response0[sns_response0.charge_mc >= thr]
        else:
            sns_response0 = sns_response0[sns_response0.ToT>0]

    except OSError:
        print(f'File {filename} does not exist')
        continue

    df_sns_resp = pd.concat([df_sns_resp, sns_response0], ignore_index=False, sort=False)


## Coincidences:
df_coinc = filter_coincidences(df_sns_resp)

## Coincidences + Centered events Hamamatsu
df_sns_resp_coinc_cent = select_evts_with_max_charge_at_center(df_coinc, variable=variable)
perc_ch_corona = get_perc_ch_corona(df_sns_resp_coinc_cent, variable=variable)
df_sns_resp_coinc_cent = df_sns_resp_coinc_cent.set_index(['event_id'])
#perc_cor --> ratio
df_sns_resp_coinc_cent['ratio'] = perc_ch_corona[df_sns_resp_coinc_cent.index].values

np.savez(evt_file, event_id   =df_sns_resp_coinc_cent.index,
                   sensor_id  =df_sns_resp_coinc_cent.sensor_id,
                   charge_data=df_sns_resp_coinc_cent.charge_data,
                   charge_conv=df_sns_resp_coinc_cent.charge_conv,
                   charge_mc  =df_sns_resp_coinc_cent.charge_mc,
                   ToT        =df_sns_resp_coinc_cent.ToT,
                   ratio      =df_sns_resp_coinc_cent.ratio)
print(datetime.datetime.now())
