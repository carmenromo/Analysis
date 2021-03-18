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


print(datetime.datetime.now())

arguments     = pbf.parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
thr_ch_start  = arguments.thr_ch_start
thr_ch_nsteps = arguments.thr_ch_nsteps
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

true_z_all       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x_all        = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge_all       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts_all         = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

true_z_area0     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x_area0      = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge_area0     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts_area0       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

true_z_max_id_89 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x_max_id_89  = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge_max_id_89 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts_max_id_89   = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

sensor_corner_tile5 = 89
area0 = [8, 28, 37, 57]

evt_file   = f'{out_path}/pet_box_build_z_map_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

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
        sel_neg_phot = sel_phot0[sel_phot0<0]
        sel_pos_phot = sel_phot0[sel_phot0>0]

        sel_he0    = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_he0[sel_he0<0]
        sel_pos_he = sel_he0[sel_he0>0]

        if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_neg_he)>0:
                continue

            for th in range(thr_ch_start, thr_ch_nsteps):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
                ids_neg, pos_neg, qs_neg = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                ids_pos, pos_pos, qs_pos = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                if sum(qs_neg) == 0:
                    continue

                pos_xs = np.array(pos_neg.T[0])
                mean_x = np.average(pos_xs, weights=qs_neg)
                var_xs = np.average((pos_xs - mean_x)**2, weights=qs_neg)

                true_z_all[th].append(sel_neg_phot[0])
                var_x_all [th].append(var_xs)
                charge_all[th].append(sum(qs_neg))
                evts_all  [th].append(evt)

                max_charge_s_id = ids_neg[np.argmax(qs_neg)]
                if max_charge_s_id in area0:
                    true_z_area0[th].append(sel_neg_phot[0])
                    var_x_area0 [th].append(var_xs)
                    charge_area0[th].append(sum(qs_neg))
                    evts_area0  [th].append(evt)

                if sum(qs_pos) == 0:
                    continue
                max_charge_s_id_tile5 = ids_pos[np.argmax(qs_pos)]
                if max_charge_s_id_tile5 == sensor_corner_tile5:
                    true_z_max_id_89[th].append(sel_neg_phot[0])
                    var_x_max_id_89 [th].append(var_xs)
                    charge_max_id_89[th].append(sum(qs_neg))
                    evts_max_id_89  [th].append(evt)

true_z_all_a       = np.array([np.array(i) for i in true_z_all])
var_x_all_a        = np.array([np.array(i) for i in var_x_all])
charge_all_a       = np.array([np.array(i) for i in charge_all])
evts_all_a         = np.array([np.array(i) for i in evts_all])
true_z_area0_a     = np.array([np.array(i) for i in true_z_area0])
var_x_area0_a      = np.array([np.array(i) for i in var_x_area0])
charge_area0_a     = np.array([np.array(i) for i in charge_area0])
evts_area0_a       = np.array([np.array(i) for i in evts_area0])
true_z_max_id_89_a = np.array([np.array(i) for i in true_z_max_id_89])
var_x_max_id_89_a  = np.array([np.array(i) for i in var_x_max_id_89])
charge_max_id_89_a = np.array([np.array(i) for i in charge_max_id_89])
evts_max_id_89_a   = np.array([np.array(i) for i in evts_max_id_89])

np.savez(evt_file, true_z_all_0=true_z_all_a[0], true_z_all_1=true_z_all_a[1], true_z_all_2=true_z_all_a[2],
                   true_z_all_3=true_z_all_a[3], true_z_all_4=true_z_all_a[4], true_z_all_5=true_z_all_a[5],
                   var_x_all_0=var_x_all_a[0], var_x_all_1=var_x_all_a[1], var_x_all_2=var_x_all_a[2],
                   var_x_all_3=var_x_all_a[3], var_x_all_4=var_x_all_a[4], var_x_all_5=var_x_all_a[5],
                   charge_all_0=charge_all_a[0], charge_all_1=charge_all_a[1], charge_all_2=charge_all_a[2],
                   charge_all_3=charge_all_a[3], charge_all_4=charge_all_a[4], charge_all_5=charge_all_a[5],
                   evts_all_0=evts_all_a[0], evts_all_1=evts_all_a[1], evts_all_2=evts_all_a[2],
                   evts_all_3=evts_all_a[3], evts_all_4=evts_all_a[4], evts_all_5=evts_all_a[5],
                   true_z_area0_0=true_z_area0_a[0], true_z_area0_1=true_z_area0_a[1], true_z_area0_2=true_z_area0_a[2],
                   true_z_area0_3=true_z_area0_a[3], true_z_area0_4=true_z_area0_a[4], true_z_area0_5=true_z_area0_a[5],
                   var_x_area0_0=var_x_area0_a[0], var_x_area0_1=var_x_area0_a[1], var_x_area0_2=var_x_area0_a[2],
                   var_x_area0_3=var_x_area0_a[3], var_x_area0_4=var_x_area0_a[4], var_x_area0_5=var_x_area0_a[5],
                   charge_area0_0=charge_area0_a[0], charge_area0_1=charge_area0_a[1], charge_area0_2=charge_area0_a[2],
                   charge_area0_3=charge_area0_a[3], charge_area0_4=charge_area0_a[4], charge_area0_5=charge_area0_a[5],
                   evts_area0_0=evts_area0_a[0], evts_area0_1=evts_area0_a[1], evts_area0_2=evts_area0_a[2],
                   evts_area0_3=evts_area0_a[3], evts_area0_4=evts_area0_a[4], evts_area0_5=evts_area0_a[5],
                   true_z_max_id_89_0=true_z_max_id_89_a[0], true_z_max_id_89_1=true_z_max_id_89_a[1], true_z_max_id_89_2=true_z_max_id_89_a[2],
                   true_z_max_id_89_3=true_z_max_id_89_a[3], true_z_max_id_89_4=true_z_max_id_89_a[4], true_z_max_id_89_5=true_z_max_id_89_a[5],
                   var_x_max_id_89_0=var_x_max_id_89_a[0], var_x_max_id_89_1=var_x_max_id_89_a[1], var_x_max_id_89_2=var_x_max_id_89_a[2],
                   var_x_max_id_89_3=var_x_max_id_89_a[3], var_x_max_id_89_4=var_x_max_id_89_a[4], var_x_max_id_89_5=var_x_max_id_89_a[5],
                   charge_max_id_89_0=charge_max_id_89_a[0], charge_max_id_89_1=charge_max_id_89_a[1], charge_max_id_89_2=charge_max_id_89_a[2],
                   charge_max_id_89_3=charge_max_id_89_a[3], charge_max_id_89_4=charge_max_id_89_a[4], charge_max_id_89_5=charge_max_id_89_a[5],
                   evts_max_id_89_0=evts_max_id_89_a[0], evts_max_id_89_1=evts_max_id_89_a[1], evts_max_id_89_2=evts_max_id_89_a[2],
                   evts_max_id_89_3=evts_max_id_89_a[3], evts_max_id_89_4=evts_max_id_89_a[4], evts_max_id_89_5=evts_max_id_89_a[5])

print(datetime.datetime.now())
