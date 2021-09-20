import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import antea.io  .mc_io            as mcio

""" To run this script
python study_phot_charge_max.py 0 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
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


evt_file   = f'{out_path}/study_phot_charge_max_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

charge_all, charge_max_sns = [[] for i in range(thr_ch_nsteps-thr_ch_start)], [[] for i in range(thr_ch_nsteps-thr_ch_start)]

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename = in_path + f'{file_name}.{number_str}.pet.h5'
    try:
        sns_response = mcio.load_mcsns_response(filename)
    except OSError:
        print(f'File {filename} does not exist')
        continue
    #print(f'file {number}')

    sns_positions = mcio.load_sns_positions    (filename)
    mcparticles   = mcio.load_mcparticles      (filename)
    mchits        = mcio.load_mchits           (filename)

    DataSiPM     = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()

    for th in range(thr_ch_start, thr_ch_nsteps):
        for evt in events:
            evt_sns   = sns_response[sns_response.event_id == evt]
            evt_parts = mcparticles [mcparticles .event_id == evt]
            evt_hits  = mchits      [mchits      .event_id == evt]

            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
            if len(evt_sns) == 0:
                continue

            ## True info
            phot, true_pos_phot   = pbf.select_phot_pet_box(evt_parts, evt_hits, he_gamma=False)
            he_gamma, true_pos_he = pbf.select_phot_pet_box(evt_parts, evt_hits, he_gamma=True)

            sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
            sel_neg_phot = sel_phot0[sel_phot0<0]
            sel_pos_phot = sel_phot0[sel_phot0>0]

            sel_he0    = np.array([pos[2] for pos in true_pos_he])
            sel_neg_he = sel_he0[sel_he0<0]
            sel_pos_he = sel_he0[sel_he0>0]

            if phot and len(sel_neg_phot)>0:
                if len(sel_neg_he)>0:
                    continue

                ids1, pos1, qs1, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if len(qs1)==0:
                    continue

                if len(ids1) > 4:
                    charge_all    [th].append(np.sum(qs1))
                    charge_max_sns[th].append(np.max(qs1))

charge_phot     = np.array([np.array(i) for i in charge_phot    ])
charge_max_phot = np.array([np.array(i) for i in charge_max_phot])

np.savez(evt_file,  charge_phot_0=charge_phot[0], charge_phot_1=charge_phot[1], charge_phot_2=charge_phot[2], charge_phot_3=charge_phot[3], charge_phot_4=charge_phot[4],
                    charge_max_phot_0=charge_max_phot[0], charge_max_phot_1=charge_max_phot[1], charge_max_phot_2=charge_max_phot[2],
                    charge_max_phot_3=charge_max_phot[3], charge_max_phot_4=charge_max_phot[4])

print(datetime.datetime.now())
