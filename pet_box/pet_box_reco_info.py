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

from antea.utils.map_functions import load_map

""" To run this script
python pet_box_reco_info.py 2500 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/z_var_x_table_pet_box_HamamatsuVUV.h5
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
"""

print(datetime.datetime.now())

arguments     = pbf.parse_args_no_ths_and_zpos(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
in_path       = arguments.in_path
file_name     = arguments.file_name
zpos_file     = arguments.zpos_file
out_path      = arguments.out_path

thr_ch_start  = 0
thr_ch_nsteps = 6
thr_charge    = 1420 #pes

area0 = [8, 28, 37, 57]
sensor_corner_tile5 = 89

evt_file   = f'{out_path}/pet_box_reco_info_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'
print(zpos_file)
Zpos = load_map(zpos_file, group="Zpos",
                           node=f"f2pes200bins",
                           x_name='Var_x',
                           y_name='Zpos',
                           u_name='ZposUncertainty')

true_x, reco_x = [[] for i in range(thr_ch_start, thr_ch_nsteps)], [[] for i in range(thr_ch_start, thr_ch_nsteps)]
true_y, reco_y = [[] for i in range(thr_ch_start, thr_ch_nsteps)], [[] for i in range(thr_ch_start, thr_ch_nsteps)]
true_z, reco_z = [[] for i in range(thr_ch_start, thr_ch_nsteps)], [[] for i in range(thr_ch_start, thr_ch_nsteps)]

sns_response_area0      = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
sns_response_peak       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
event_ids               = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
sensor_corner_tile5_max = [[] for i in range(thr_ch_start, thr_ch_nsteps)]


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

        for th in range(thr_ch_start, thr_ch_nsteps):
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
            if len(evt_sns) == 0:
                continue

            ids_neg, pos_neg, qs_neg = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
            if len(qs_neg) == 0:
                continue

            phot, true_pos_phot = pbf.select_photoelectric_pet_box(evt_parts, evt_hits)
            sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
            sel_neg_phot = sel_phot0[sel_phot0<0]


            if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition

                max_charge_s_id = ids_neg[np.argmax(qs_neg)]
                if max_charge_s_id in area0:
                    sns_response_area0[th].append(sum(qs_neg))

                    if sum(qs_neg)>1420:
                        pos_xs = np.array(pos_neg.T[0])
                        mean_x = np.average(pos_xs, weights=qs_neg)
                        var_xs = np.average((pos_xs - mean_x)**2, weights=qs_neg)

                        pos_ys = np.array(pos_neg.T[1])
                        mean_y = np.average(pos_ys, weights=qs_neg)

                        z_pos = Zpos(var_xs).value

                        true_pos_neg_evt = true_pos_phot[sel_phot0<0][0]

                        sns_response_peak[th].append(sum(qs_neg))

                        reco_x[th].append(mean_x)
                        reco_y[th].append(mean_y)
                        reco_z[th].append(z_pos)

                        true_x[th].append(true_pos_neg_evt[0])
                        true_y[th].append(true_pos_neg_evt[1])
                        true_z[th].append(true_pos_neg_evt[2])

                        event_ids[th].append(evt)

                        ids_pos, pos_pos, qs_pos = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                        if len(qs_pos)!=0:
                            max_charge_s_id_tile5 = ids_pos[np.argmax(qs_pos)]
                            if max_charge_s_id_tile5 == sensor_corner_tile5:
                                sensor_corner_tile5_max[th].append(True)
                            else:
                                sensor_corner_tile5_max[th].append(False)
                        else:
                            sensor_corner_tile5_max[th].append(False)

true_x_a = np.array([np.array(i) for i in true_x])
reco_x_a = np.array([np.array(i) for i in reco_x])
true_y_a = np.array([np.array(i) for i in true_y])
reco_y_a = np.array([np.array(i) for i in reco_y])
true_z_a = np.array([np.array(i) for i in true_z])
reco_z_a = np.array([np.array(i) for i in reco_z])

sns_response_area0_a      = np.array([np.array(i) for i in sns_response_area0])
sns_response_peak_a       = np.array([np.array(i) for i in sns_response_peak])
event_ids_a               = np.array([np.array(i) for i in event_ids])
sensor_corner_tile5_max_a = np.array([np.array(i) for i in sensor_corner_tile5_max])

np.savez(evt_file, true_x_0=true_x_a[0], true_x_1=true_x_a[1], true_x_2=true_x_a[2],
                   true_x_3=true_x_a[3], true_x_4=true_x_a[4], true_x_5=true_x_a[5],
                   reco_x_0=reco_x_a[0], reco_x_1=reco_x_a[1], reco_x_2=reco_x_a[2],
                   reco_x_3=reco_x_a[3], reco_x_4=reco_x_a[4], reco_x_5=reco_x_a[5],
                   true_y_0=true_y_a[0], true_y_1=true_y_a[1], true_y_2=true_y_a[2],
                   true_y_3=true_y_a[3], true_y_4=true_y_a[4], true_y_5=true_y_a[5],
                   reco_y_0=reco_y_a[0], reco_y_1=reco_y_a[1], reco_y_2=reco_y_a[2],
                   reco_y_3=reco_y_a[3], reco_y_4=reco_y_a[4], reco_y_5=reco_y_a[5],
                   true_z_0=true_z_a[0], true_z_1=true_z_a[1], true_z_2=true_z_a[2],
                   true_z_3=true_z_a[3], true_z_4=true_z_a[4], true_z_5=true_z_a[5],
                   reco_z_0=reco_z_a[0], reco_z_1=reco_z_a[1], reco_z_2=reco_z_a[2],
                   reco_z_3=reco_z_a[3], reco_z_4=reco_z_a[4], reco_z_5=reco_z_a[5],
                   sns_response_area0_0=sns_response_area0_a[0], sns_response_area0_1=sns_response_area0_a[1],
                   sns_response_area0_2=sns_response_area0_a[2], sns_response_area0_3=sns_response_area0_a[3],
                   sns_response_area0_4=sns_response_area0_a[4], sns_response_area0_5=sns_response_area0_a[5],
                   sns_response_peak_0=sns_response_peak_a[0], sns_response_peak_1=sns_response_peak_a[1],
                   sns_response_peak_2=sns_response_peak_a[2], sns_response_peak_3=sns_response_peak_a[3],
                   sns_response_peak_4=sns_response_peak_a[4], sns_response_peak_5=sns_response_peak_a[5],
                   event_ids_0=event_ids_a[0], event_ids_1=event_ids_a[1], event_ids_2=event_ids_a[2],
                   event_ids_3=event_ids_a[3], event_ids_4=event_ids_a[4], event_ids_5=event_ids_a[5],
                   sensor_corner_tile5_max_0=sensor_corner_tile5_max_a[0], sensor_corner_tile5_max_1=sensor_corner_tile5_max_a[1],
                   sensor_corner_tile5_max_2=sensor_corner_tile5_max_a[2], sensor_corner_tile5_max_3=sensor_corner_tile5_max_a[3],
                   sensor_corner_tile5_max_4=sensor_corner_tile5_max_a[4], sensor_corner_tile5_max_5=sensor_corner_tile5_max_a[5])

print(datetime.datetime.now())
