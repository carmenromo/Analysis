import sys
import math
import argparse
import tables as tb
import numpy  as np
import pandas as pd
import datetime

import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

""" To run this script
python pet_box_build_z_map_coinc_plane.py 0 1 0 6 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV
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

true_z = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x  = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts   = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

sensor_corner_tile5 = 89

evt_file   = f'{out_path}/pet_box_build_z_map_coinc_plane_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

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

        phot, true_pos_phot   = pbf.select_photoelectric_pet_box(evt_parts, evt_hits)
        he_gamma, true_pos_he = pbf.select_gamma_high_energy(evt_parts, evt_hits)

        sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
        sel_pos_phot = sel_phot0[sel_phot0>0]

        sel_he0    = np.array([pos[2] for pos in true_pos_he])
        sel_pos_he = sel_he0[sel_he0>0]

        if phot and len(sel_pos_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_pos_he)>0:
                continue

            for th in range(thr_ch_start, thr_ch_nsteps):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
                ids_pos, pos_pos, qs_pos = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                if sum(qs_pos) == 0:
                    continue

                max_charge_s_id_tile5 = ids_pos[np.argmax(qs_pos)]
                if max_charge_s_id_tile5 == sensor_corner_tile5:
                    print(evt)
                    pos_xs = np.array(pos_pos.T[0])
                    mean_x = np.average(pos_xs, weights=qs_pos)
                    var_xs = np.average((pos_xs - mean_x)**2, weights=qs_pos)

                    true_z[th].append(sel_pos_phot[0])
                    var_x [th].append(var_xs)
                    charge[th].append(sum(qs_pos))
                    evts  [th].append(evt)


true_z_a = np.array([np.array(i) for i in true_z])
var_x_a  = np.array([np.array(i) for i in var_x])
charge_a = np.array([np.array(i) for i in charge])
evts_a   = np.array([np.array(i) for i in evts])


np.savez(evt_file, true_z_0=true_z_a[0], true_z_1=true_z_a[1], true_z_2=true_z_a[2], true_z_3=true_z_a[3], true_z_4=true_z_a[4],
         true_z_5=true_z_a[5], var_x_0=var_x_a[0], var_x_1=var_x_a[1], var_x_2=var_x_a[2], var_x_3=var_x_a[3],
         var_x_4=var_x_a[4], var_x_5=var_x_a[5], charge_0=charge_a[0], charge_1=charge_a[1], charge_2=charge_a[2],
         charge_3=charge_a[3], charge_4=charge_a[4], charge_5=charge_a[5], evts_0=evts_a[0], evts_1=evts_a[1],
         evts_2=evts_a[2], evts_3=evts_a[3], evts_4=evts_a[4], evts_5=evts_a[5])

print(datetime.datetime.now())
