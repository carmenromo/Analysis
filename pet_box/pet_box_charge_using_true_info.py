import sys
import argparse
import datetime
import tables   as tb
import numpy    as np
import pandas   as pd

import pet_box_functions as pbf

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.reco_functions as rf

""" To run this script
python pet_box_charge_using_true_info.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

evt_file   = f'{out_path}/pet_box_charge_true_info_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

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

    mcparticles   = load_mcparticles   (true_file)
    mchits        = load_mchits        (true_file)
    sns_response  = load_mcsns_response(true_file)
    sns_positions = load_sns_positions (true_file)
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]

        phot, _ = pbf.select_phot_pet_box(evt_parts, evt_hits)
        if phot:
            for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=n_th)
                if len(evt_sns) == 0:
                    continue
                ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                tot_charges[n_th].append(sum(qs))
            evt_ids.append(evt)
        else:
            continue

tot_charges = np.array(tot_charges)
evt_ids     = np.array(evt_ids)

np.savez(evt_file, tot_charges=tot_charges, evt_ids=evt_ids)

print(datetime.datetime.now())
