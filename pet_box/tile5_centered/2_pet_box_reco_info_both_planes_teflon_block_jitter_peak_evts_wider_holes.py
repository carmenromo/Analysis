import sys
import argparse
import datetime

import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.reco.reco_functions    as rf
import antea.elec.shaping_functions as shf
import antea.reco.mctrue_functions  as mcf
import antea.io  .mc_io             as mcio

import antea.database.load_db              as db
import antea.mcsim   .sensor_functions     as snsf

from antea.utils.map_functions import load_map
from invisible_cities.core     import system_of_units as units

""" To run this script
python 2_pet_box_reco_info_both_planes_teflon_block_jitter.py 0 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_HamamatsuVUV
 teflon_block /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/tile5_centered/data_reco_info
"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file', type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'   , type = int, help = "number of files to analize")
    parser.add_argument('in_path'   ,             help = "input files path"          )
    parser.add_argument('file_name' ,             help = "name of input files"       )
    parser.add_argument('peak_min'  , type = int, help = "Minimum of photopeak"      )
    parser.add_argument('peak_max'  , type = int, help = "Maximum of photopeak"      )
    parser.add_argument('out_name' ,              help = "name of output files"      )
    parser.add_argument('out_path'  ,             help = "output files path"         )
    return parser.parse_args()
print(datetime.datetime.now())

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
in_path       = arguments.in_path
file_name     = arguments.file_name
#zpos_file     = arguments.zpos_file
#zpos_file2    = arguments.zpos_file2
peak_min      = arguments.peak_min
peak_max      = arguments.peak_max
out_name      = arguments.out_name
out_path      = arguments.out_path

# int_area = np.array([22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77,
#                      33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66, 44, 45, 54, 55])

evt_file   = f'{out_path}/pet_box_reco_info_HamVUV_both_planes_fluct_jitter_peak_evts_{peak_min}_{peak_max}_{out_name}_{start}_{numb}'

# Zpos = load_map(zpos_file, group="Zpos",
#                             node=f"f2pes200bins",
#                             x_name='Var_x',
#                             y_name='Zpos',
#                             u_name='ZposUncertainty')


timestamp_thr = [0, 0.25, 0.50, 0.75]
### parameters for single photoelectron convolution in SiPM response
tau_sipm       = [100, 15000]
time_window    = 5000
time           = np.arange(0, 5000)
spe_resp, norm = shf.normalize_sipm_shaping(time, tau_sipm)

sigma_sipm = 40 #ps
sigma_elec = 30 #ps

reco_x1, reco_x2 = [], []
reco_y1, reco_y2 = [], []
reco_z1, reco_z2 = [], []

sns_resp1, sns_resp2 = [], []
max_resp1, max_resp2 = [], []

first_sipm1, first_sipm2 = [[] for i in range(len(timestamp_thr))], [[] for i in range(len(timestamp_thr))]
first_time1, first_time2 = [[] for i in range(len(timestamp_thr))], [[] for i in range(len(timestamp_thr))]

event_ids1 = []
event_ids2 = []
event_ids_times = []

DataSiPM_pb     = db.DataSiPM('petalo', 12406, 'PB')
DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    #filename = in_path + f'{file_name}.{number_str}.pet.h5'
    filename = in_path + f'{file_name}.{number}.h5'
    try:
        sns_response = mcio.load_mcsns_response(filename)
    except OSError:
        print(f'File {filename} does not exist')
        continue
    except KeyError:
        print(f'File {filename} KeyError')
        continue
    #print(f'file {number}')

    #tof_bin_size  = mcio.read_sensor_bin_width_from_conf(filename, tof=True)
    sns_positions = mcio.load_sns_positions    (filename)
    tof_response  = mcio.load_mcTOFsns_response(filename)

    #DataSiPM     = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    #DataSiPM_idx = DataSiPM.set_index('SensorID')

    events = sns_response.event_id.unique()
    th     = 2
    for evt in events:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_tof = tof_response[tof_response.event_id == evt]

        #times = evt_tof.time_bin.values * tof_bin_size / units.ps
        times = np.round(evt_tof.time.values / units.ps)
        ## INTRINSIC SIPM FLUCTUATIONS
        if sigma_sipm > 0:
            times = np.round(np.random.normal(times, sigma_sipm))
        #     evt_tof.insert(len(evt_tof.columns), 'time', np.round(np.random.normal(times, sigma_sipm)).astype(int))
        # else:
        #     evt_tof.insert(len(evt_tof.columns), 'time', times.astype(int))

        evt_tof = evt_tof.drop('time', axis=1) # drop original time
        evt_tof.insert(len(evt_tof.columns), 'time', times.astype(int)) # round to 1 ps
        evt_tof.insert(len(evt_tof.columns), 'charge', np.ones(len(times)).astype(int)) # add 1 unit of charge per time

        fluct_sns_response = snsf.apply_charge_fluctuation(evt_sns, DataSiPM_pb_idx)
        evt_sns = rf.find_SiPMs_over_threshold(fluct_sns_response, threshold=th)
        if len(evt_sns) == 0:
            continue

        ids_over_thr = evt_sns.sensor_id.astype('int64').values
        #evt_tof      = evt_tof[evt_tof.sensor_id.isin(-ids_over_thr)]
        evt_tof      = evt_tof[evt_tof.sensor_id.isin(evt_sns.sensor_id)]

        ids1, pos1, qs1, ids2, pos2, qs2 = pbf.info_from_the_tiles(DataSiPM_pb_idx, evt_sns)
        if len(qs1)==0 or len(qs2)==0:
            continue
        if max(qs1) < peak_min or max(qs2) < peak_min:
            continue
        if max(qs1) > peak_max or max(qs2) > peak_max:
            continue

        print(f'Passed event: {evt}')

        # max_charge_s_id       = ids1[np.argmax(qs1)]
        # max_charge_s_id_tile5 = ids2[np.argmax(qs2)]
        #
        # if (max_charge_s_id in int_area) and (max_charge_s_id_tile5 in int_area+100):
        sns_resp1.append(sum(qs1))
        max_resp1.append(max(qs1))

        pos_xs1 = np.array(pos1.T[0])
        mean_x1 = np.average(pos_xs1, weights=qs1)
        var_xs1 = np.average((pos_xs1 - mean_x1)**2, weights=qs1)

        pos_ys1 = np.array(pos1.T[1])
        mean_y1 = np.average(pos_ys1, weights=qs1)

        # z_pos1 = Zpos(var_xs1).value

        reco_x1.append(mean_x1)
        reco_y1.append(mean_y1)
        #reco_z1.append(-53.25)
        reco_z1.append(-36.65)
        event_ids1.append(evt)


        sns_resp2.append(sum(qs2))
        max_resp2.append(max(qs2))

        pos_xs2 = np.array(pos2.T[0])
        mean_x2 = np.average(pos_xs2, weights=qs2)
        var_xs2 = np.average((pos_xs2 - mean_x2)**2, weights=qs2)

        pos_ys2 = np.array(pos2.T[1])
        mean_y2 = np.average(pos_ys2, weights=qs2)

        #### Because planes are symmetric!!
        #z_pos2 = -Zpos(var_xs2).value

        reco_x2.append(mean_x2)
        reco_y2.append(mean_y2)
        #reco_z2.append(53.25)
        reco_z2.append(36.65)

        event_ids2.append(evt)

        ## produce a TOF dataframe with convolved time response
        tof_sns = evt_tof.sensor_id.unique()
        evt_tof_exp_dist = []
        for s_id in tof_sns:
            tdc_conv    = shf.sipm_shaping_convolution(evt_tof, spe_resp, s_id, time_window)
            tdc_conv_df = shf.build_convoluted_df(evt, s_id, tdc_conv)
            if sigma_elec != 0:
                tdc_conv_df.assign(time=np.random.normal(tdc_conv_df.time.values, sigma_elec))
            evt_tof_exp_dist.append(tdc_conv_df)

        if len(evt_tof_exp_dist) == 0:
            continue
        else:
            evt_tof_exp_dist = pd.concat(evt_tof_exp_dist)

        ## Calculate different thresholds in charge
        for k, th in enumerate(timestamp_thr):
            evt_tof_exp_dist = evt_tof_exp_dist[evt_tof_exp_dist.charge > th/norm]
            min_id1, min_id2, min_t1, min_t2 = rf.find_coincidence_timestamps(evt_tof_exp_dist, ids1, ids2)
            print(evt_tof_exp_dist)
            # try:
            #     min_id1, min_id2, min_t1, min_t2 = rf.find_coincidence_timestamps(evt_tof_exp_dist, ids1, ids2)
            # except Exception as e:
            #     print(e)
            #     print("Function find_coincidence_timestamps fails, event = ", evt)
            #     min_id1, min_id2, min_t1, min_t2 = [-1], [-1], -1, -1

            first_sipm1[k].append(min_id1[0])
            first_time1[k].append(min_t1)

            first_sipm2[k].append(min_id2[0])
            first_time2[k].append(min_t2)
        event_ids_times.append(evt)

reco_x1 = np.array(reco_x1)
reco_x2 = np.array(reco_x2)
reco_y1 = np.array(reco_y1)
reco_y2 = np.array(reco_y2)
reco_z1 = np.array(reco_z1)
reco_z2 = np.array(reco_z2)

sns_resp1 = np.array(sns_resp1)
sns_resp2 = np.array(sns_resp2)

max_resp1 = np.array(max_resp1)
max_resp2 = np.array(max_resp2)

event_ids1 = np.array(event_ids1)
event_ids2 = np.array(event_ids2)
event_ids_times = np.array(event_ids_times)

first_sipm1 = np.array([np.array(i) for i in first_sipm1])
first_sipm2 = np.array([np.array(i) for i in first_sipm2])
first_time1 = np.array([np.array(i) for i in first_time1])
first_time2 = np.array([np.array(i) for i in first_time2])

np.savez(evt_file, reco_x1=reco_x1, reco_x2=reco_x2, reco_y1=reco_y1, reco_y2=reco_y2, reco_z1=reco_z1, reco_z2=reco_z2,
                   sns_resp1=sns_resp1, sns_resp2=sns_resp2, max_resp1=max_resp1, max_resp2=max_resp2,
                   event_ids1=event_ids1, event_ids2=event_ids2, event_ids_times=event_ids_times,
                   first_sipm1_0=first_sipm1[0], first_sipm2_0=first_sipm2[0], first_time1_0=first_time1[0], first_time2_0=first_time2[0],
                   first_sipm1_1=first_sipm1[1], first_sipm2_1=first_sipm2[1], first_time1_1=first_time1[1], first_time2_1=first_time2[1],
                   first_sipm1_2=first_sipm1[2], first_sipm2_2=first_sipm2[2],first_time1_2=first_time1[2], first_time2_2=first_time2[2],
                   first_sipm1_3=first_sipm1[3], first_sipm2_3=first_sipm2[3], first_time1_3=first_time1[3], first_time2_3=first_time2[3])

print(datetime.datetime.now())
