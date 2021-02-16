import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import pet_box_functions as pbf

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.reco_functions as rf

""" To run thie script
python pet_box_charge_selecting_areas.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

evt_file   = f'{out_path}/pet_box_charge_select_areas_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

chargs_a0 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_a1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_a2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_a3 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_a4 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_a5 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evt_ids   = []

area0 = [8, 28, 37, 57]
area1 = [7, 15, 16, 19, 20, 27, 38, 45, 46, 49, 50, 58]
area2 = [6, 10, 11, 12, 14, 18, 22, 23, 24, 26, 39, 41, 42, 43, 47, 51, 53, 54, 55, 59]
area3 = [1, 2, 3, 4, 5, 9, 13, 17, 21, 25, 29, 30, 31, 32, 33, 34, 35, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]
area4 = area0 + area1
area5 = area0 + area1 + area2

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
        for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
            #max_charge_s_id = evt_sns[evt_sns.charge == evt_sns.charge.max()].sensor_id.values[0]
            if len(evt_sns) == 0:
                continue
            ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
            if len(qs) == 0:
                continue
            max_charge_s_id = ids[np.argmax(qs)]
            if max_charge_s_id in area0:
                chargs_a0[n_th].append(sum(qs))
            elif max_charge_s_id in area1:
                chargs_a1[n_th].append(sum(qs))
            elif max_charge_s_id in area2:
                chargs_a2[n_th].append(sum(qs))
            elif max_charge_s_id in area3:
                chargs_a3[n_th].append(sum(qs))
            if max_charge_s_id in area4:
                chargs_a4[n_th].append(sum(qs))
            if max_charge_s_id in area5:
                chargs_a5[n_th].append(sum(qs))
        evt_ids.append(evt)

chargs_a0 = np.array(chargs_a0)
chargs_a1 = np.array(chargs_a1)
chargs_a2 = np.array(chargs_a2)
chargs_a3 = np.array(chargs_a3)
chargs_a4 = np.array(chargs_a4)
chargs_a5 = np.array(chargs_a5)
evt_ids   = np.array(evt_ids)

np.savez(evt_file, chargs_a0=chargs_a0, chargs_a1=chargs_a1, chargs_a2=chargs_a2,
                   chargs_a3=chargs_a3, chargs_a4=chargs_a4, chargs_a5=chargs_a5, evt_ids=evt_ids)

print(datetime.datetime.now())
