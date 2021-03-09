import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import pet_box_functions as pbf

from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf

""" To run this script
python pet_box_charge_selecting_area0_true_info_dists.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV
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

area0 = [8, 28, 37, 57]
threshold = 2

evt_file   = f'{out_path}/reco_pet_box_coord_using_the_two_phot_peaks_{start}_{numb}_thr{threshold}pes'

tot_charges0 = []
tot_charges1 = []
var_r0       = []
var_r1       = []
true_z0      = []
true_z1      = []
evt_ids0     = []
evt_ids1     = []

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
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]
        evt_sns   = sns_response[sns_response.event_id == evt]

        he_gamma, true_pos_he = pbf.select_gamma_high_energy(evt_parts, evt_hits)
        phot, true_pos_phot   = mcf.select_photoelectric(evt_parts, evt_hits)

        sel_phot0    = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_phot0[sel_phot0<0]
        phot_neg_pos = np.array(true_pos_phot)[sel_phot0<0]

        sel_he0    = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_he0[sel_he0<0]

        if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_neg_he)>0: ### Be careful with the meaning of this condition
                continue
            else:
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                if len(evt_sns) == 0:
                    continue

                ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                if len(qs) == 0:
                    continue

                max_charge_s_id = ids[np.argmax(qs)]
                if max_charge_s_id in area0:
                    pos_xs = np.array(pos.T[0])
                    pos_ys = np.array(pos.T[1])
                    pos_zs = np.array(pos.T[2])

                    pos_r  = np.array([np.sqrt(p[0]**2 + p[1]**2) for p in pos])
                    mean_r = np.average(pos_r, weights=qs)
                    var_rs = np.average((pos_r - mean_r)**2, weights=qs)

                    if sum(qs) < 1370:
                        var_r0      .append(var_rs)
                        true_z0     .append(sel_neg_phot[0])
                        tot_charges0.append(sum(qs))
                        evt_ids0    .append(evt)
                    else:
                        var_r1      .append(var_rs)
                        true_z1     .append(sel_neg_phot[0])
                        tot_charges1.append(sum(qs))
                        evt_ids1    .append(evt)
        else:
            continue

tot_charges0 = np.array(tot_charges0)
tot_charges1 = np.array(tot_charges1)
evt_ids0     = np.array(evt_ids0)
evt_ids1     = np.array(evt_ids1)
var_r0       = np.array(var_r0 )
true_z0      = np.array(true_z0)
var_r1       = np.array(var_r1 )
true_z1      = np.array(true_z1)

np.savez(evt_file, tot_charges0=tot_charges0, tot_charges1=tot_charges1, evt_ids0=evt_ids0, evt_ids1=evt_ids1,
var_r0=var_r0, true_z0=true_z0, var_r1=var_r1, true_z1=true_z1)

print(datetime.datetime.now())
