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
python 1_pet_box_build_z_map_both_planes.py 0 1 0 6 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/
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

true_z1_all     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x1_all      = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge1_all     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
max_charge1_all = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts1_all       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

true_z1     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x1      = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge1     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
max_charge1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts1       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

true_z2_all     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x2_all      = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge2_all     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
max_charge2_all = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts2_all       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

true_z2     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_x2      = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
charge2     = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
max_charge2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evts2       = [[] for i in range(thr_ch_start, thr_ch_nsteps)]


int_area = np.array([22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77,
                     33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66, 44, 45, 54, 55])

evt_file   = f'{out_path}/pet_box_true_info_teflon_block_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

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
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]

        phot, true_pos_phot = mcf.select_photoelectric(evt_parts, evt_hits)

        sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_phot0[sel_phot0<0]
        sel_pos_phot = sel_phot0[sel_phot0>0]

        if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
            for th in range(thr_ch_start, thr_ch_nsteps):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
                ids1, pos1, qs1, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if sum(qs1) == 0:
                    continue

                pos_xs = np.array(pos1.T[0])
                mean_x = np.average(pos_xs, weights=qs1)
                var_xs = np.average((pos_xs - mean_x)**2, weights=qs1)

                true_z1_all    [th].append(sel_neg_phot[0])
                var_x1_all     [th].append(var_xs)
                charge1_all    [th].append(sum(qs1))
                max_charge1_all[th].append(max(qs1))
                evts1_all      [th].append(evt)

                max_charge_s_id = ids1[np.argmax(qs1)]
                if max_charge_s_id in int_area:
                    true_z1    [th].append(sel_neg_phot[0])
                    var_x1     [th].append(var_xs)
                    charge1    [th].append(sum(qs1))
                    max_charge1[th].append(max(qs1))
                    evts1      [th].append(evt)

        elif phot and len(sel_pos_phot)>0: ### Be careful with the meaning of this condition
            for th in range(thr_ch_start, thr_ch_nsteps):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
                _, _, _, ids2, pos2, qs2 = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if sum(qs2) == 0:
                    continue

                pos_xs = np.array(pos2.T[0])
                mean_x = np.average(pos_xs, weights=qs2)
                var_xs = np.average((pos_xs - mean_x)**2, weights=qs2)

                true_z2_all    [th].append(sel_pos_phot[0])
                var_x2_all     [th].append(var_xs)
                charge2_all    [th].append(sum(qs2))
                max_charge2_all[th].append(max(qs2))
                evts2_all      [th].append(evt)

                max_charge_s_id = ids2[np.argmax(qs2)]
                if max_charge_s_id in int_area+100:
                    true_z2    [th].append(sel_pos_phot[0])
                    var_x2     [th].append(var_xs)
                    charge2    [th].append(sum(qs2))
                    max_charge2[th].append(max(qs2))
                    evts2      [th].append(evt)

true_z1_all_a = np.array([np.array(i) for i in true_z1_all])
var_x1_all_a  = np.array([np.array(i) for i in var_x1_all ])
charge1_all_a = np.array([np.array(i) for i in charge1_all])
evts1_all_a   = np.array([np.array(i) for i in evts1_all  ])
true_z1_a     = np.array([np.array(i) for i in true_z1    ])
var_x1_a      = np.array([np.array(i) for i in var_x1     ])
charge1_a     = np.array([np.array(i) for i in charge1    ])
evts1_a       = np.array([np.array(i) for i in evts1      ])
true_z2_all_a = np.array([np.array(i) for i in true_z2_all])
var_x2_all_a  = np.array([np.array(i) for i in var_x2_all ])
charge2_all_a = np.array([np.array(i) for i in charge2_all])
evts2_all_a   = np.array([np.array(i) for i in evts2_all  ])
true_z2_a     = np.array([np.array(i) for i in true_z2    ])
var_x2_a      = np.array([np.array(i) for i in var_x2     ])
charge2_a     = np.array([np.array(i) for i in charge2    ])
evts2_a       = np.array([np.array(i) for i in evts2      ])

max_charge1_all_a = np.array([np.array(i) for i in max_charge1_all])
max_charge1_a     = np.array([np.array(i) for i in max_charge1    ])
max_charge2_all_a = np.array([np.array(i) for i in max_charge2_all])
max_charge2_a     = np.array([np.array(i) for i in max_charge2    ])


np.savez(evt_file, true_z1_all_0=true_z1_all_a[0], true_z1_all_1=true_z1_all_a[1], true_z1_all_2=true_z1_all_a[2],
        true_z1_all_3=true_z1_all_a[3], true_z1_all_4=true_z1_all_a[4], true_z1_all_5=true_z1_all_a[5],
        var_x1_all_0=var_x1_all_a[0], var_x1_all_1=var_x1_all_a[1], var_x1_all_2=var_x1_all_a[2],
        var_x1_all_3=var_x1_all_a[3], var_x1_all_4=var_x1_all_a[4], var_x1_all_5=var_x1_all_a[5],
        charge1_all_0=charge1_all_a[0], charge1_all_1=charge1_all_a[1], charge1_all_2=charge1_all_a[2],
        charge1_all_3=charge1_all_a[3], charge1_all_4=charge1_all_a[4], charge1_all_5=charge1_all_a[5],
        evts1_all_0=evts1_all_a[0], evts1_all_1=evts1_all_a[1], evts1_all_2=evts1_all_a[2],
        evts1_all_3=evts1_all_a[3], evts1_all_4=evts1_all_a[4], evts1_all_5=evts1_all_a[5],
        true_z1_0=true_z1_a[0], true_z1_1=true_z1_a[1], true_z1_2=true_z1_a[2],
        true_z1_3=true_z1_a[3], true_z1_4=true_z1_a[4], true_z1_5=true_z1_a[5],
        var_x1_0=var_x1_a[0], var_x1_1=var_x1_a[1], var_x1_2=var_x1_a[2],
        var_x1_3=var_x1_a[3], var_x1_4=var_x1_a[4], var_x1_5=var_x1_a[5],
        charge1_0=charge1_a[0], charge1_1=charge1_a[1], charge1_2=charge1_a[2],
        charge1_3=charge1_a[3], charge1_4=charge1_a[4], charge1_5=charge1_a[5],
        evts1_0=evts1_a[0], evts1_1=evts1_a[1], evts1_2=evts1_a[2],
        evts1_3=evts1_a[3], evts1_4=evts1_a[4], evts1_5=evts1_a[5],
        true_z2_all_0=true_z2_all_a[0], true_z2_all_1=true_z2_all_a[1], true_z2_all_2=true_z2_all_a[2],
        true_z2_all_3=true_z2_all_a[3], true_z2_all_4=true_z2_all_a[4], true_z2_all_5=true_z2_all_a[5],
        var_x2_all_0=var_x2_all_a[0], var_x2_all_1=var_x2_all_a[1], var_x2_all_2=var_x2_all_a[2],
        var_x2_all_3=var_x2_all_a[3], var_x2_all_4=var_x2_all_a[4], var_x2_all_5=var_x2_all_a[5],
        charge2_all_0=charge2_all_a[0], charge2_all_1=charge2_all_a[1], charge2_all_2=charge2_all_a[2],
        charge2_all_3=charge2_all_a[3], charge2_all_4=charge2_all_a[4], charge2_all_5=charge2_all_a[5],
        evts2_all_0=evts2_all_a[0], evts2_all_1=evts2_all_a[1], evts2_all_2=evts2_all_a[2],
        evts2_all_3=evts2_all_a[3], evts2_all_4=evts2_all_a[4], evts2_all_5=evts2_all_a[5],
        true_z2_0=true_z2_a[0], true_z2_1=true_z2_a[1], true_z2_2=true_z2_a[2],
        true_z2_3=true_z2_a[3], true_z2_4=true_z2_a[4], true_z2_5=true_z2_a[5],
        var_x2_0=var_x2_a[0], var_x2_1=var_x2_a[1], var_x2_2=var_x2_a[2],
        var_x2_3=var_x2_a[3], var_x2_4=var_x2_a[4], var_x2_5=var_x2_a[5],
        charge2_0=charge2_a[0], charge2_1=charge2_a[1], charge2_2=charge2_a[2],
        charge2_3=charge2_a[3], charge2_4=charge2_a[4], charge2_5=charge2_a[5],
        evts2_0=evts2_a[0], evts2_1=evts2_a[1], evts2_2=evts2_a[2],
        evts2_3=evts2_a[3], evts2_4=evts2_a[4], evts2_5=evts2_a[5],
        max_charge1_all_0=max_charge1_all[0], max_charge1_all_1=max_charge1_all[1],
        max_charge1_all_2=max_charge1_all[2], max_charge1_all_3=max_charge1_all[3],
        max_charge1_all_4=max_charge1_all[4], max_charge1_all_5=max_charge1_all[5],
        max_charge1_0=max_charge1[0], max_charge1_1=max_charge1[1], max_charge1_2=max_charge1[2],
        max_charge1_3=max_charge1[3], max_charge1_4=max_charge1[4], max_charge1_5=max_charge1[5],
        max_charge2_all_0=max_charge2_all[0], max_charge2_all_1=max_charge2_all[1],
        max_charge2_all_2=max_charge2_all[2], max_charge2_all_3=max_charge2_all[3],
        max_charge2_all_4=max_charge2_all[4], max_charge2_all_5=max_charge2_all[5],
        max_charge2_0=max_charge2[0], max_charge2_1=max_charge2[1], max_charge2_2=max_charge2[2],
        max_charge2_3=max_charge2[3], max_charge2_4=max_charge2[4], max_charge2_5=max_charge2[5])

print(datetime.datetime.now())
