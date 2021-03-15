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
#thr_ch_start  = arguments.thr_ch_start
#thr_ch_nsteps = arguments.thr_ch_nsteps
thr_ch_start  = 0
thr_ch_nsteps = 6
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

sensor_corner_tile5 = 89
area0 = [8, 28, 37, 57]
area1 = [7, 15, 16, 19, 20, 27, 38, 45, 46, 49, 50, 58]
area2 = [6, 10, 11, 12, 14, 18, 22, 23, 24, 26, 39, 41, 42, 43, 47, 51, 53, 54, 55, 59]
area3 = [1, 2, 3, 4, 5, 9, 13, 17, 21, 25, 29, 30, 31, 32, 33, 34, 35, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]

threshold = 2

evt_file   = f'{out_path}/pet_box_charge_selecting_corner_tile5_no_select_areas_{start}_{numb}_thr{threshold}pes'

num_evt_max_id_tile5_corner = 0
#num_evt_max_id_tiles_area0  = 0
#num_evt_max_id_tiles_area1  = 0
#num_evt_max_id_tiles_area2  = 0
#num_evt_max_id_tiles_area3  = 0

#charge0, charge1, charge2, charge3, charge4 = [], [], [], [], []
charge_tile5     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge_det_plane = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge_max_id_89 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
tot_evts = 0

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
        tot_evts += 1
        evt_sns = sns_response[sns_response.event_id == evt]
        for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
            if len(evt_sns) == 0:
                continue
            ids_pos, pos_pos, qs_pos = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
            ids_neg, pos_neg, qs_neg = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)

            if len(qs_pos)==0 or len(qs_neg)==0:
                continue

            max_charge_s_id_pos = ids_pos[np.argmax(qs_pos)]
            max_charge_s_id_neg = ids_neg[np.argmax(qs_neg)]

            if max_charge_s_id_pos == sensor_corner_tile5:
                num_evt_max_id_tile5_corner += 1
                charge_tile5    [n_th].append(sum(qs_pos))
                charge_det_plane[n_th].append(sum(qs_neg))
                charge_max_id_89[n_th].append(max(qs_pos))

            # charge4.append(sum(qs_pos))
            # if max_charge_s_id_neg in area0:
            #     num_evt_max_id_tiles_area0 += 1
            #     charge0.append(sum(qs_neg))
            # elif max_charge_s_id_neg in area1:
            #     num_evt_max_id_tiles_area1 += 1
            #     charge1.append(sum(qs_neg))
            # elif max_charge_s_id_neg in area2:
            #     num_evt_max_id_tiles_area2 += 1
            #     charge2.append(sum(qs_neg))
            # elif max_charge_s_id_neg in area3:
            #     num_evt_max_id_tiles_area3 += 1
            #     charge3.append(sum(qs_neg))

charge_tile5_0 = np.array(charge_tile5[0])
charge_tile5_1 = np.array(charge_tile5[1])
charge_tile5_2 = np.array(charge_tile5[2])
charge_tile5_3 = np.array(charge_tile5[3])
charge_tile5_4 = np.array(charge_tile5[4])
charge_tile5_5 = np.array(charge_tile5[5])

charge_det_plane_0 = np.array(charge_det_plane[0])
charge_det_plane_1 = np.array(charge_det_plane[1])
charge_det_plane_2 = np.array(charge_det_plane[2])
charge_det_plane_3 = np.array(charge_det_plane[3])
charge_det_plane_4 = np.array(charge_det_plane[4])
charge_det_plane_5 = np.array(charge_det_plane[5])

charge_max_id_89_0 = charge_max_id_89[0]
charge_max_id_89_1 = charge_max_id_89[1]
charge_max_id_89_2 = charge_max_id_89[2]
charge_max_id_89_3 = charge_max_id_89[3]
charge_max_id_89_4 = charge_max_id_89[4]
charge_max_id_89_5 = charge_max_id_89[5]

# charge0 = np.array(charge0)
# charge1 = np.array(charge1)
# charge2 = np.array(charge2)
# charge3 = np.array(charge3)
# charge4 = np.array(charge4)

print(charge_tile5_5)
print(charge_det_plane_5)
np.savez(evt_file, charge_tile5_0=charge_tile5_0, charge_tile5_1=charge_tile5_1, charge_tile5_2=charge_tile5_2,
         charge_tile5_3=charge_tile5_3, charge_tile5_4=charge_tile5_4, charge_tile5_5=charge_tile5_5,
         charge_det_plane_0=charge_det_plane_0, charge_det_plane_1=charge_det_plane_1, charge_det_plane_2=charge_det_plane_2,
         charge_det_plane_3=charge_det_plane_3, charge_det_plane_4=charge_det_plane_4, charge_det_plane_5=charge_det_plane_5,
         charge_max_id_89_0=charge_max_id_89_0, charge_max_id_89_1=charge_max_id_89_1, charge_max_id_89_2=charge_max_id_89_2,
         charge_max_id_89_3=charge_max_id_89_3, charge_max_id_89_4=charge_max_id_89_4, charge_max_id_89_5=charge_max_id_89_5
         num_evt_max_id_tile5_corner=num_evt_max_id_tile5_corner, tot_evts=tot_evts)
# np.savez(evt_file, charge0=charge0, charge1=charge1, charge2=charge2, charge3=charge3, charge4=charge4,
#          num_evt_max_id_tile5_corner=num_evt_max_id_tile5_corner, num_evt_max_id_tiles_area0=num_evt_max_id_tiles_area0,
#          num_evt_max_id_tiles_area1=num_evt_max_id_tiles_area1, num_evt_max_id_tiles_area2=num_evt_max_id_tiles_area2,
#          num_evt_max_id_tiles_area3=num_evt_max_id_tiles_area3, tot_evts=tot_evts)

print(datetime.datetime.now())
