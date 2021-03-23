import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import pet_box_functions as pbf

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf

""" To run this script
python pet_box_charge_selecting_areas_each_tile.py 0 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

sensor_corner_tile5 = 89
area0 = [8, 28, 37, 57]
area1 = [7, 15, 16, 19, 20, 27, 38, 45, 46, 49, 50, 58]
area2 = [6, 10, 11, 12, 14, 18, 22, 23, 24, 26, 39, 41, 42, 43, 47, 51, 53, 54, 55, 59]
area3 = [1, 2, 3, 4, 5, 9, 13, 17, 21, 25, 29, 30, 31, 32, 33, 34, 35, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]
area4 = area0 + area1
area5 = area4 + area2
area6 = area5 + area3


threshold = 2

evt_file   = f'{out_path}/pet_box_charge_selecting_corner_tile5_events_covered_{start}_{numb}_thr{threshold}pes'

num_areas        = 4
charge_tile5     = [[] for i in range(num_areas)]
charge_det_plane = [[] for i in range(num_areas)]
evt_ids          = [[] for i in range(num_areas+1)]

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = f'{in_path}/{file_name}.{number_str}.pet.h5'
    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        print(f'File {true_file} does not exist')
        continue
    print(f'Analyzing file {true_file}')

    mcparticles   = load_mcparticles   (true_file)
    sns_response  = load_mcsns_response(true_file)
    sns_positions = load_sns_positions (true_file)
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
        if len(evt_sns) == 0:
            continue
        ids_pos, pos_pos, qs_pos = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
        ids_neg, pos_neg, qs_neg = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)

        if len(qs_neg)==0:
            continue

        if set(ids_neg).issubset(set(area0)):
            charge_tile5    [0].append(sum(qs_pos))
            charge_det_plane[0].append(sum(qs_neg))
            evt_ids         [0].append(evt)
        if set(ids_neg).issubset(set(area4)):
            charge_tile5    [1].append(sum(qs_pos))
            charge_det_plane[1].append(sum(qs_neg))
            evt_ids         [1].append(evt)
        if set(ids_neg).issubset(set(area5)):
            charge_tile5    [2].append(sum(qs_pos))
            charge_det_plane[2].append(sum(qs_neg))
            evt_ids         [2].append(evt)
        if set(ids_neg).issubset(set(area6)):
            charge_tile5    [3].append(sum(qs_pos))
            charge_det_plane[3].append(sum(qs_neg))
            evt_ids         [3].append(evt)

        if len(qs_pos)==0:
            continue
        max_charge_s_id_pos = ids_pos[np.argmax(qs_pos)]
        if max_charge_s_id_pos == sensor_corner_tile5:
            evt_ids[4].append(evt)

charge_tile5_0 = np.array(charge_tile5[0])
charge_tile5_1 = np.array(charge_tile5[1])
charge_tile5_2 = np.array(charge_tile5[2])
charge_tile5_3 = np.array(charge_tile5[3])

charge_det_plane_0 = np.array(charge_det_plane[0])
charge_det_plane_1 = np.array(charge_det_plane[1])
charge_det_plane_2 = np.array(charge_det_plane[2])
charge_det_plane_3 = np.array(charge_det_plane[3])

evt_ids_0 = np.array(evt_ids[0])
evt_ids_1 = np.array(evt_ids[1])
evt_ids_2 = np.array(evt_ids[2])
evt_ids_3 = np.array(evt_ids[3])
evt_ids_4 = np.array(evt_ids[4])

np.savez(evt_file, charge_tile5_0=charge_tile5_0, charge_tile5_1=charge_tile5_1, charge_tile5_2=charge_tile5_2, charge_tile5_3=charge_tile5_3,
         charge_det_plane_0=charge_det_plane_0, charge_det_plane_1=charge_det_plane_1, charge_det_plane_2=charge_det_plane_2, charge_det_plane_3=charge_det_plane_3,
         evt_ids_0=evt_ids_0, evt_ids_1=evt_ids_1, evt_ids_2=evt_ids_2, evt_ids_3=evt_ids_3, evt_ids_4=evt_ids_4)

print(datetime.datetime.now())
