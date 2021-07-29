import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import pet_box_functions as pbf

import antea.io.mc_io as mcio

import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf

""" To run this script
python pet_box_charge_FBK_events_covered.py 1590 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_tile5centered_FBK
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

area0 = [44, 45, 54, 55]
area1 = [33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66]
area2 = [22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77]
area3 = [11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
         51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88]
area4 = area0 + area1
area5 = area0 + area1 + area2

area0_tile5 = [122, 123, 132, 133]

threshold = 2

evt_file   = f'{out_path}/pet_box_charge_FBK_events_covered_{start}_{numb}_thr{threshold}pes'

num_areas        = 3
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

    mcparticles   = mcio.load_mcparticles   (true_file)
    sns_response  = mcio.load_mcsns_response(true_file)
    sns_positions = mcio.load_sns_positions (true_file)

    ## FBK!!!
    sns_positions_c = pbf.correct_FBK_sensor_pos(sns_positions, both_planes=False)
    DataSiPM        = sns_positions_c.rename(columns={"sensor_id": "SensorID","new_x": "X", "new_y": "Y", "z": "Z"})
    #DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
        if len(evt_sns) == 0:
            continue
        ids1, pos1, qs1, ids2, pos2, qs2 = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)

        if len(qs1)==0:
            continue

        if set(ids1).issubset(set(area0)):
            charge_tile5    [0].append(sum(qs2))
            charge_det_plane[0].append(sum(qs1))
            evt_ids         [0].append(evt)
        if set(ids1).issubset(set(area4)):
            charge_tile5    [1].append(sum(qs2))
            charge_det_plane[1].append(sum(qs1))
            evt_ids         [1].append(evt)
        if set(ids1).issubset(set(area5)):
            charge_tile5    [2].append(sum(qs2))
            charge_det_plane[2].append(sum(qs1))
            evt_ids         [2].append(evt)

        if len(qs2)==0:
            continue

        max_charge_s_id2 = ids2[np.argmax(qs2)]
        if max_charge_s_id2 in area0_tile5:
            evt_ids[3].append(evt)

charge_tile5_0 = np.array(charge_tile5[0])
charge_tile5_1 = np.array(charge_tile5[1])
charge_tile5_2 = np.array(charge_tile5[2])

charge_det_plane_0 = np.array(charge_det_plane[0])
charge_det_plane_1 = np.array(charge_det_plane[1])
charge_det_plane_2 = np.array(charge_det_plane[2])

evt_ids_0 = np.array(evt_ids[0])
evt_ids_1 = np.array(evt_ids[1])
evt_ids_2 = np.array(evt_ids[2])
evt_ids_3 = np.array(evt_ids[3])

np.savez(evt_file, charge_tile5_0=charge_tile5_0, charge_tile5_1=charge_tile5_1, charge_tile5_2=charge_tile5_2,
         charge_det_plane_0=charge_det_plane_0, charge_det_plane_1=charge_det_plane_1, charge_det_plane_2=charge_det_plane_2,
         evt_ids_0=evt_ids_0, evt_ids_1=evt_ids_1, evt_ids_2=evt_ids_2, evt_ids_3=evt_ids_3)

print(datetime.datetime.now())
