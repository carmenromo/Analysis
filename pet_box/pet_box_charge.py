import sys
import argparse
import datetime
import tables   as tb
import numpy    as np
import pandas   as pd

import pet_box_functions as pbf

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.reco_functions as rf

""" To run thie script
python pet_box_charge.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

evt_file   = f'{out_path}/pet_box_charge_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

tot_charges = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evt_ids     = []

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = f'{in_path}/{file_name}.{number_str}.pet.h5'
    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        print(f'File {true_file} does not exist')
        continue
    print(f'Analyzing file {true_file}')

    particles     = load_mcparticles   (true_file)
    sns_response  = load_mcsns_response(true_file)
    sns_positions = load_sns_positions (true_file)
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = particles.event_id.unique()
    for evt in events:
        evt_sns = sns_response[sns_response.event_id == evt]
        for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=n_th)
            if len(evt_sns) == 0:
                continue

            sipms        = DataSiPM_idx.loc[evt_sns.sensor_id]
            sns_ids      = sipms.index.astype('int64').values
            sns_pos      = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
            sns_charges  = evt_sns.charge
            sel          = sipms.Z.values<0 #Plane with high number of sensors
            sns, pos, qs = sns_ids[sel], sns_pos[sel], sns_charges[sel]
            tot_charges[n_th].append(sum(qs))
        evt_ids.append(evt)

tot_charges = np.array(tot_charges)
evt_ids     = np.array(evt_ids)

np.savez(evt_file, tot_charges=tot_charges, evt_ids=evt_ids)

print(datetime.datetime.now())
