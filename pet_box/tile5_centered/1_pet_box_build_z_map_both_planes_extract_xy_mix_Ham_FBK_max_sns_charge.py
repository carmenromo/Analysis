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

import antea.io.mc_io as mcio

""" To run this script
python 1_pet_box_build_z_map_both_planes_extract_xy_mix_Ham_FBK.py 0 1 0 6 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/
PetBox_asymmetric_tile5centered_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

true_z1_all = []
true_x1_all = []
true_y1_all = []
var_x1_all  = []
charge1_all = []
max_charge1_all = []
evts1_all   = []

true_z1 = []
true_x1 = []
true_y1 = []
var_x1  = []
charge1 = []
max_charge1 = []
evts1   = []

true_z2_all = []
true_x2_all = []
true_y2_all = []
var_x2_all  = []
charge2_all = []
max_charge2_all = []
evts2_all   = []

true_z2 = []
true_x2 = []
true_y2 = []
var_x2  = []
charge2 = []
max_charge2 = []
evts2   = []


area0       = [ 44,  45,  54,  55]
area0_tile5 = [122, 123, 132, 133]

evt_file   = f'{out_path}/pet_box_build_z_map_both_planes_mix_Ham_FBK_max_ch_sns_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

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
    mchits        = mcio.load_mchits        (true_file)
    sns_response  = mcio.load_mcsns_response(true_file)
    sns_positions = mcio.load_sns_positions (true_file)
    sns_positions_c = pbf.correct_FBK_sensor_pos(sns_positions, both_planes=False)
    DataSiPM      = sns_positions_c.rename(columns={"sensor_id": "SensorID","new_x": "X", "new_y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]

        phot, true_pos_phot   = pbf.select_phot_pet_box(evt_parts, evt_hits, he_gamma=False)
        he_gamma, true_pos_he = pbf.select_phot_pet_box(evt_parts, evt_hits, he_gamma=True)

        sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_phot0[sel_phot0<0]
        sel_pos_phot = sel_phot0[sel_phot0>0]

        sel_phot0_x  = np.array([pos[0] for pos in true_pos_phot])
        sel_neg_phot_x = sel_phot0_x[sel_phot0<0]
        sel_pos_phot_x = sel_phot0_x[sel_phot0>0]
        sel_phot0_y  = np.array([pos[1] for pos in true_pos_phot])
        sel_neg_phot_y = sel_phot0_y[sel_phot0<0]
        sel_pos_phot_y = sel_phot0_y[sel_phot0>0]

        sel_he0    = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_he0[sel_he0<0]
        sel_pos_he = sel_he0[sel_he0>0]

        if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_neg_he)>0:
                continue

            #for th in range(thr_ch_start, thr_ch_nsteps):
            th = 2
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
            ids1, pos1, qs1, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
            if sum(qs1) == 0:
                continue

            pos_xs = np.array(pos1.T[0])
            mean_x = np.average(pos_xs, weights=qs1)
            var_xs = np.average((pos_xs - mean_x)**2, weights=qs1)

            true_z1_all.append(sel_neg_phot[0])
            true_x1_all.append(sel_neg_phot_x[0])
            true_y1_all.append(sel_neg_phot_y[0])
            var_x1_all .append(var_xs)
            charge1_all.append(sum(qs1))
            max_charge1_all.append(max(qs1))
            evts1_all  .append(evt)

            max_charge_s_id = ids1[np.argmax(qs1)]
            if max_charge_s_id in area0:
                true_z1.append(sel_neg_phot[0])
                true_x1.append(sel_neg_phot_x[0])
                true_y1.append(sel_neg_phot_y[0])
                var_x1 .append(var_xs)
                charge1.append(sum(qs1))
                max_charge1.append(max(qs1))
                evts1  .append(evt)

        elif phot and len(sel_pos_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_pos_he)>0:
                continue

            #for th in range(thr_ch_start, thr_ch_nsteps):
            th = 2
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
            _, _, _, ids2, pos2, qs2 = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
            if sum(qs2) == 0:
                continue

            pos_xs = np.array(pos2.T[0])
            mean_x = np.average(pos_xs, weights=qs2)
            var_xs = np.average((pos_xs - mean_x)**2, weights=qs2)

            true_z2_all.append(sel_pos_phot[0])
            true_x2_all.append(sel_pos_phot_x[0])                
            true_y2_all.append(sel_pos_phot_y[0])
            var_x2_all .append(var_xs)
            charge2_all.append(sum(qs2))
            max_charge2_all.append(max(qs2))
            evts2_all  .append(evt)

            max_charge_s_id = ids2[np.argmax(qs2)]
            if max_charge_s_id in area0_tile5:
                true_z2.append(sel_pos_phot[0])
                true_x2.append(sel_pos_phot_x[0])
                true_y2.append(sel_pos_phot_y[0])
                var_x2 .append(var_xs)
                charge2.append(sum(qs2))
                max_charge2.append(max(qs2))
                evts2  .append(evt)

true_z1_all_a = np.array(true_z1_all)
true_x1_all_a = np.array(true_x1_all)
true_y1_all_a = np.array(true_y1_all)
var_x1_all_a  = np.array(var_x1_all )
charge1_all_a = np.array(charge1_all)
max_charge1_all_a = np.array(max_charge1_all)
evts1_all_a   = np.array(evts1_all  )

true_z1_a     = np.array(true_z1    )
true_x1_a     = np.array(true_x1    )
true_y1_a     = np.array(true_y1    )
var_x1_a      = np.array(var_x1     )
charge1_a     = np.array(charge1    )
max_charge1_a = np.array(max_charge1)
evts1_a       = np.array(evts1      )

true_z2_all_a = np.array(true_z2_all)
true_x2_all_a = np.array(true_x2_all)
true_y2_all_a = np.array(true_y2_all)
var_x2_all_a  = np.array(var_x2_all )
charge2_all_a = np.array(charge2_all)
max_charge2_all_a = np.array(max_charge2_all)
evts2_all_a   = np.array(evts2_all  )

true_z2_a     = np.array(true_z2    )
true_x2_a     = np.array(true_x2    )
true_y2_a     = np.array(true_y2    )
var_x2_a      = np.array(var_x2     )
charge2_a     = np.array(charge2    )
max_charge2_a = np.array(max_charge2)
evts2_a       = np.array(evts2      )


np.savez(evt_file, true_z1_all=true_z1_all_a, true_x1_all=true_x1_all_a, true_y1_all=true_y1_all_a, var_x1_all=var_x1_all_a, 
         charge1_all=charge1_all_a, max_charge1_all=max_charge1_all_a, evts1_all=evts1_all_a, true_z1=true_z1_a, true_x1=true_x1_a, 
         true_y1=true_y1_a, var_x1=var_x1_a, charge1=charge1_a, max_charge1=max_charge1_a, evts1=evts1_a, true_z2_all=true_z2_all_a, 
         true_x2_all=true_x2_all_a, true_y2_all=true_y2_all_a, var_x2_all=var_x2_all_a, charge2_all=charge2_all_a, 
         max_charge2_all=max_charge2_all_a, evts2_all=evts2_all_a, true_z2=true_z2_a, true_x2=true_x2_a, true_y2=true_y2_a, 
         var_x2=var_x2_a, charge2=charge2_a, max_charge2=max_charge2_a, evts2=evts2_a)

print(datetime.datetime.now())
