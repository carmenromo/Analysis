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
python pet_box_charge_studing_he_gamma_events.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

evt_file  = f'{out_path}/pet_box_charge_studying_he_gamma_evts_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'
threshold = 2

evt_ids_phot_and_he_gamma    = []
sig_phot0                    = []
charge_phot0                 = []
sig_gamma0                   = []
charge_gamma0                = []
evt_ids_phot_and_no_he_gamma = []
sig_phot1                    = []
charge_phot1                 = []
evt_ids_no_phot_and_he_gamma = []
sig_gamma1                   = []
charge_gamma1                = []

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
        phot, true_pos_phot0  = mcf.select_photoelectric(evt_parts, evt_hits)

        true_pos_phot = []
        for p1 in true_pos_he:
            for p2 in true_pos_phot0:
                if p1[2]==p2[2]:
                    continue
                else:
                    true_pos_phot.append(p2)

        true_pos_phot = np.array(true_pos_phot)
        if len(true_pos_phot)==0:
            phot = False
        else:
            phot = True

        sel_neg_phot0 = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot  = sel_neg_phot0[sel_neg_phot0<0]

        sel_neg_he0 = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_neg_he0[sel_neg_he0<0]

        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
        if phot:
            if he_gamma:
                evt_ids_phot_and_he_gamma.append(evt)
                if len(sel_neg_phot)>0:
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    sig_phot0   .append(True)
                    charge_phot0.append(sum(qs))
                else:
                    ids, pos, qs = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                    sig_phot0   .append(False)
                    charge_phot0.append(sum(qs))

                if len(sel_neg_he)>0:
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    sig_gamma0   .append(True)
                    charge_gamma0.append(sum(qs))
                else:
                    ids, pos, qs = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                    sig_gamma0   .append(False)
                    charge_gamma0.append(sum(qs))
            else:
                evt_ids_phot_and_no_he_gamma.append(evt)
                if len(sel_neg_phot)>0:
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    sig_phot1   .append(True)
                    charge_phot1.append(sum(qs))
                else:
                    ids, pos, qs = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                    sig_phot1   .append(False)
                    charge_phot1.append(sum(qs))

        else:
            if he_gamma:
                evt_ids_no_phot_and_he_gamma.append(evt)
                if len(sel_neg_he)>0:
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    sig_gamma1   .append(True)
                    charge_gamma1.append(sum(qs))
                else:
                    ids, pos, qs = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                    sig_gamma1   .append(False)
                    charge_gamma1.append(sum(qs))

evt_ids_phot_and_he_gamma    = np.array(evt_ids_phot_and_he_gamma)
evt_ids_phot_and_no_he_gamma = np.array(evt_ids_phot_and_no_he_gamma)
evt_ids_no_phot_and_he_gamma = np.array(evt_ids_no_phot_and_he_gamma)

sig_phot0  = np.array(sig_phot0)
sig_gamma0 = np.array(sig_gamma0)
sig_phot1  = np.array(sig_phot1)
sig_gamma1 = np.array(sig_gamma1)

charge_phot0  = np.array(charge_phot0)
charge_gamma0 = np.array(charge_gamma0)
charge_phot1  = np.array(charge_phot1)
charge_gamma1 = np.array(charge_gamma1)

np.savez(evt_file, evt_ids_phot_and_he_gamma=evt_ids_phot_and_he_gamma, evt_ids_phot_and_no_he_gamma=evt_ids_phot_and_no_he_gamma,
         evt_ids_no_phot_and_he_gamma=evt_ids_no_phot_and_he_gamma, sig_phot0=sig_phot0, sig_gamma0=sig_gamma0, sig_phot1=sig_phot1,
         sig_gamma1=sig_gamma1, charge_phot0=charge_phot0, charge_gamma0=charge_gamma0, charge_phot1=charge_phot1, charge_gamma1=charge_gamma1)

print(datetime.datetime.now())
