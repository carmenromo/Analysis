import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.elec.tof_functions    as tf
import antea.reco.mctrue_functions as mcf
import antea.io  .mc_io            as mcio

from antea.utils.map_functions import load_map
from invisible_cities.core     import system_of_units as units

""" To run this script
python pet_box_reco_info_both_planes.py 2500 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/z_var_x_table_pet_box_HamamatsuVUV.h5
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/z_var_x_table_coinc_plane_pet_box_HamamatsuVUV.h5
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
"""

print(datetime.datetime.now())

arguments     = pbf.parse_args_no_ths_and_zpos2(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
in_path       = arguments.in_path
file_name     = arguments.file_name
zpos_file     = arguments.zpos_file
zpos_file2    = arguments.zpos_file2
out_path      = arguments.out_path

thr_ch_start  = 0
thr_ch_nsteps = 6
thr_charge1   = 1420 #pes
thr_charge2   = 150 #pes

area0 = [8, 28, 37, 57]
sensor_corner_tile5 = 89

evt_file   = f'{out_path}/pet_box_reco_info_both_planes_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

Zpos1 = load_map(zpos_file, group="Zpos",
                            node=f"f2pes200bins",
                            x_name='Var_x',
                            y_name='Zpos',
                            u_name='ZposUncertainty')
Zpos2 = load_map(zpos_file2, group="Zpos",
                             node=f"f2pes200bins",
                             x_name='Var_x',
                             y_name='Zpos',
                             u_name='ZposUncertainty')

timestamp_thr = [0, 0.25, 0.50, 0.75]
### parameters for single photoelectron convolution in SiPM response
tau_sipm       = [100, 15000]
time_window    = 10000
time_bin       = 5 # ps
time           = np.arange(0, 80000, time_bin)
time           = time + (time_bin/2)
spe_resp, norm = tf.apply_spe_dist(time, tau_sipm)


reco_x1, reco_x2 = [], []
reco_y1, reco_y2 = [], []
reco_z1, reco_z2 = [], []

true_x1, true_x2 = [], []
true_y1, true_y2 = [], []
true_z1, true_z2 = [], []

sns_resp1, sns_resp2 = [], []

first_sipm1, first_sipm2 = [[] for i in range(len(timestamp_thr))], [[] for i in range(len(timestamp_thr))]
first_time1, first_time2 = [[] for i in range(len(timestamp_thr))], [[] for i in range(len(timestamp_thr))]

event_ids       = []
event_ids_times = []

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename = in_path + f'PetBox_asymmetric_HamamatsuVUV.{number_str}.pet.h5'
    try:
        sns_response = mcio.load_mcsns_response(filename)
    except OSError:
        print(f'File {filename} does not exist')
        continue
    #print(f'file {number}')

    tof_bin_size = mcio.read_sensor_bin_width_from_conf(filename, tof=True)

    sns_positions = mcio.load_sns_positions    (filename)
    mcparticles   = mcio.load_mcparticles      (filename)
    mchits        = mcio.load_mchits           (filename)
    tof_response  = mcio.load_mcTOFsns_response(filename)

    DataSiPM     = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    th     = 2
    for evt in events[:]:
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]
        evt_tof   = tof_response[tof_response.event_id == evt]


        ## True info
        phot, true_pos_phot   = pbf.select_photoelectric_pet_box(evt_parts, evt_hits)
        he_gamma, true_pos_he = pbf.select_gamma_high_energy(evt_parts, evt_hits)

        sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_phot0[sel_phot0<0]
        sel_pos_phot = sel_phot0[sel_phot0>0]

        sel_he0    = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_he0[sel_he0<0]
        sel_pos_he = sel_he0[sel_he0>0]

        if phot and len(sel_neg_phot)>0 and len(sel_pos_phot)>0: ### coincidences
            if len(sel_neg_he)>0 or len(sel_pos_he)>0:
                continue

            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
            if len(evt_sns) == 0:
                continue

            ids1, pos1, qs1, ids2, pos2, qs2 = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
            if len(qs1)==0 or len(qs2)==0:
                continue

            max_charge_s_id       = ids1[np.argmax(qs1)]
            max_charge_s_id_tile5 = ids2[np.argmax(qs2)]
            if max_charge_s_id in area0 and max_charge_s_id_tile5==sensor_corner_tile5:

                sns_resp1.append(sum(qs1))
                sns_resp2.append(sum(qs2))

                true_pos_neg_evt = true_pos_phot[sel_phot0<0][0]
                true_pos_pos_evt = true_pos_phot[sel_phot0>0][0]

                if sum(qs1)>1420:
                    pos_xs1 = np.array(pos1.T[0])
                    mean_x1 = np.average(pos_xs1, weights=qs1)
                    var_xs1 = np.average((pos_xs1 - mean_x1)**2, weights=qs1)

                    pos_ys1 = np.array(pos1.T[1])
                    mean_y1 = np.average(pos_ys1, weights=qs1)

                    z_pos1 = Zpos1(var_xs1).value

                    reco_x1.append(mean_x1)
                    reco_y1.append(mean_y1)
                    reco_z1.append(z_pos1)

                    true_x1.append(true_pos_neg_evt[0])
                    true_y1.append(true_pos_neg_evt[1])
                    true_z1.append(true_pos_neg_evt[2])
                else:
                    reco_x1.append(None)
                    reco_y1.append(None)
                    reco_z1.append(None)
                    true_x1.append(None)
                    true_y1.append(None)
                    true_z1.append(None)

                if sum(qs2)>150:
                    pos_xs2 = np.array(pos2.T[0])
                    mean_x2 = np.average(pos_xs2, weights=qs2)
                    var_xs2 = np.average((pos_xs2 - mean_x2)**2, weights=qs2)

                    pos_ys2 = np.array(pos2.T[1])
                    mean_y2 = np.average(pos_ys2, weights=qs2)

                    z_pos2 = Zpos2(var_xs2).value

                    reco_x2.append(mean_x2)
                    reco_y2.append(mean_y2)
                    reco_z2.append(z_pos2)

                    true_x2.append(true_pos_pos_evt[0])
                    true_y2.append(true_pos_pos_evt[1])
                    true_z2.append(true_pos_pos_evt[2])

                else:
                    reco_x2.append(None)
                    reco_y2.append(None)
                    reco_z2.append(None)
                    true_x2.append(None)
                    true_y2.append(None)
                    true_z2.append(None)

                event_ids.append(evt)

                if sum(qs1)>1420 and sum(qs2)>150:
                    ## produce a TOF dataframe with convolved time response
                    tof_sns = evt_tof.sensor_id.unique()
                    evt_tof_exp_dist = []
                    for s_id in tof_sns:
                        tdc_conv    = tf.tdc_convolution(evt_tof, spe_resp, s_id, time_window)
                        tdc_conv_df = tf.translate_charge_conv_to_wf_df(evt, s_id, tdc_conv)
                        evt_tof_exp_dist.append(tdc_conv_df)
                    evt_tof_exp_dist = pd.concat(evt_tof_exp_dist)

                    ## Calculate different thresholds in charge
                    for k, th in enumerate(timestamp_thr):
                        evt_tof_exp_dist = evt_tof_exp_dist[evt_tof_exp_dist.charge > th/norm]
                        try:
                            min_id1, min_id2, min_t1, min_t2 = rf.find_coincidence_timestamps(evt_tof_exp_dist, ids1, ids2)
                        except:
                            min_id1, min_id2, min_t1, min_t2 = -1, -1, -1, -1

                        first_sipm1[k].append(min_id1)
                        first_time1[k].append(min_t1*tof_bin_size/units.ps)

                        first_sipm2[k].append(min_id2)
                        first_time2[k].append(min_t2*tof_bin_size/units.ps)

                    event_ids_times.append(evt)

reco_x1 = np.array(reco_x1)
reco_x2 = np.array(reco_x2)
reco_y1 = np.array(reco_y1)
reco_y2 = np.array(reco_y2)
reco_z1 = np.array(reco_z1)
reco_z2 = np.array(reco_z2)

true_x1 = np.array(true_x1)
true_x2 = np.array(true_x2)
true_y1 = np.array(true_y1)
true_y2 = np.array(true_y2)
true_z1 = np.array(true_z1)
true_z2 = np.array(true_z2)

sns_resp1       = np.array(sns_resp1)
sns_resp2       = np.array(sns_resp2)
event_ids       = np.array(event_ids)
event_ids_times = np.array(event_ids_times)

first_sipm1 = np.array([np.array(i) for i in first_sipm1])
first_sipm2 = np.array([np.array(i) for i in first_sipm2])
first_time1 = np.array([np.array(i) for i in first_time1])
first_time2 = np.array([np.array(i) for i in first_time2])

np.savez(evt_file, reco_x1=reco_x1, reco_x2=reco_x2, reco_y1=reco_y1, reco_y2=reco_y2, reco_z1=reco_z1, reco_z2=reco_z2,
                   true_x1=true_x1, true_x2=true_x2, true_y1=true_y1, true_y2=true_y2, true_z1=true_z1, true_z2=true_z2,
                   sns_resp1=sns_resp1, sns_resp2=sns_resp2, event_ids=event_ids, event_ids_times=event_ids_times,
                   first_sipm1_0=first_sipm1[0], first_sipm2_0=first_sipm2[0], first_time1_0=first_time1[0], first_time2_0=first_time2[0],
                   first_sipm1_1=first_sipm1[1], first_sipm2_1=first_sipm2[1], first_time1_1=first_time1[1], first_time2_1=first_time2[1],
                   first_sipm1_2=first_sipm1[2], first_sipm2_2=first_sipm2[2],first_time1_2=first_time1[2], first_time2_2=first_time2[2],
                   first_sipm1_3=first_sipm1[3], first_sipm2_3=first_sipm2[3], first_time1_3=first_time1[3], first_time2_3=first_time2[3])

print(datetime.datetime.now())
