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

evt_file   = f'{out_path}/pet_box_charge_select_area0_true_info_dists_saveqs_{start}_{numb}_thr{threshold}pes'

tot_charges           = []
dist_ztrue_zsens      = []
dist_true_sens_module = []
all_true_pos          = []
all_charges           = []
all_sns_positions     = []
all_sns_ids           = []
touched_sipms         = []
evt_ids               = []

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

        sel_neg_phot0 = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_neg_phot0[sel_neg_phot0<0]
        phot_neg_pos = np.array(true_pos_phot)[sel_neg_phot0<0]

        sel_neg_he = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_neg_he[sel_neg_he<0]

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

                    dtz_sens  = phot_neg_pos[0][2] - (-55.405)
                    sel_sens  = sns_positions[sns_positions.sensor_id == max_charge_s_id]
                    sens_pos  = np.array([sel_sens.x.values[0], sel_sens.y.values[0], sel_sens.z.values[0]])
                    dtz_truep = np.linalg.norm(np.subtract(phot_neg_pos[0], sens_pos))

                    tot_charges          .append(sum(qs))
                    dist_ztrue_zsens     .append(dtz_sens)
                    dist_true_sens_module.append(dtz_truep)
                    all_true_pos         .append(phot_neg_pos[0])
                    all_charges          .append(qs)
                    all_sns_positions    .append(pos)
                    all_sns_ids          .append(ids)
                    touched_sipms        .append(len(qs))
                    evt_ids              .append(evt)
        else:
            continue

tot_charges           = np.array(tot_charges          )
dist_ztrue_zsens      = np.array(dist_ztrue_zsens     )
dist_true_sens_module = np.array(dist_true_sens_module)
all_true_pos          = np.array(all_true_pos)
all_charges           = np.array(all_charges)
all_sns_positions     = np.array(all_sns_positions)
all_sns_ids           = np.array(all_sns_ids)
touched_sipms         = np.array(touched_sipms        )
evt_ids               = np.array(evt_ids              )

print(all_true_pos)
np.savez(evt_file, tot_charges=tot_charges, dist_ztrue_zsens=dist_ztrue_zsens, dist_true_sens_module=dist_true_sens_module, all_true_pos=all_true_pos,
         all_charges=all_charges, all_sns_positions=all_sns_positions, all_sns_ids=all_sns_ids, touched_sipms=touched_sipms, evt_ids=evt_ids)

print(datetime.datetime.now())
