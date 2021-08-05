import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import pet_box_functions as pbf

import antea.io.mc_io as mcio

import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf

""" To run this script
python pet_box_charge_Hamamatsu_true_info.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV
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

evt_file   = f'{out_path}/pet_box_charge_Hamamatsu_true_info_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

# phot + he_gamma
chargs_phot_he_gamma_a0 = []
chargs_phot_he_gamma_a1 = []
chargs_phot_he_gamma_a2 = []
chargs_phot_he_gamma_a3 = []
chargs_phot_he_gamma_a4 = []
chargs_phot_he_gamma_a5 = []

# phot + NO he_gamma
chargs_phot_no_he_gamma_a0 = []
chargs_phot_no_he_gamma_a1 = []
chargs_phot_no_he_gamma_a2 = []
chargs_phot_no_he_gamma_a3 = []
chargs_phot_no_he_gamma_a4 = []
chargs_phot_no_he_gamma_a5 = []

# NO phot + he_gamma
chargs_no_phot_he_gamma_a0 = []
chargs_no_phot_he_gamma_a1 = []
chargs_no_phot_he_gamma_a2 = []
chargs_no_phot_he_gamma_a3 = []
chargs_no_phot_he_gamma_a4 = []
chargs_no_phot_he_gamma_a5 = []

# NO phot + NO he_gamma
chargs_no_phot_no_he_gamma_a0 = []
chargs_no_phot_no_he_gamma_a1 = []
chargs_no_phot_no_he_gamma_a2 = []
chargs_no_phot_no_he_gamma_a3 = []
chargs_no_phot_no_he_gamma_a4 = []
chargs_no_phot_no_he_gamma_a5 = []
evt_ids   = []

area0 = [44, 45, 54, 55]
area1 = [33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66]
area2 = [22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77]
area3 = [11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
         51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88]
area4 = area0 + area1
area5 = area0 + area1 + area2

area0_tile5 = [122, 123, 132, 133]

threshold = 2

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
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]
        evt_sns   = sns_response[sns_response.event_id == evt]

        phot, true_pos_phot   = pbf.select_phot_pet_box(evt_parts, evt_hits, he_gamma=False)
        he_gamma, true_pos_he = pbf.select_phot_pet_box(evt_parts, evt_hits, he_gamma=True)

        sel_neg_phot = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_neg_phot[sel_neg_phot<0]

        sel_neg_he = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_neg_he[sel_neg_he<0]

        if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_neg_he)>0: ### Be careful with the meaning of this condition
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                if len(evt_sns) == 0:
                    continue
                ids, pos, qs, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if len(qs) == 0:
                    continue
                max_charge_s_id = ids[np.argmax(qs)]
                if   max_charge_s_id in area0: chargs_phot_he_gamma_a0.append(sum(qs))
                elif max_charge_s_id in area1: chargs_phot_he_gamma_a1.append(sum(qs))
                elif max_charge_s_id in area2: chargs_phot_he_gamma_a2.append(sum(qs))
                elif max_charge_s_id in area3: chargs_phot_he_gamma_a3.append(sum(qs))
                if   max_charge_s_id in area4: chargs_phot_he_gamma_a4.append(sum(qs))
                if   max_charge_s_id in area5: chargs_phot_he_gamma_a5.append(sum(qs))
            else:
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                if len(evt_sns) == 0:
                    continue
                ids, pos, qs, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if len(qs) == 0:
                    continue
                max_charge_s_id = ids[np.argmax(qs)]
                if   max_charge_s_id in area0: chargs_phot_no_he_gamma_a0.append(sum(qs))
                elif max_charge_s_id in area1: chargs_phot_no_he_gamma_a1.append(sum(qs))
                elif max_charge_s_id in area2: chargs_phot_no_he_gamma_a2.append(sum(qs))
                elif max_charge_s_id in area3: chargs_phot_no_he_gamma_a3.append(sum(qs))
                if   max_charge_s_id in area4: chargs_phot_no_he_gamma_a4.append(sum(qs))
                if   max_charge_s_id in area5: chargs_phot_no_he_gamma_a5.append(sum(qs))
        else:
            if he_gamma and len(sel_neg_he)>0: ### Be careful with the meaning of this condition
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                if len(evt_sns) == 0:
                    continue
                ids, pos, qs, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if len(qs) == 0:
                    continue
                max_charge_s_id = ids[np.argmax(qs)]
                if   max_charge_s_id in area0: chargs_no_phot_he_gamma_a0.append(sum(qs))
                elif max_charge_s_id in area1: chargs_no_phot_he_gamma_a1.append(sum(qs))
                elif max_charge_s_id in area2: chargs_no_phot_he_gamma_a2.append(sum(qs))
                elif max_charge_s_id in area3: chargs_no_phot_he_gamma_a3.append(sum(qs))
                if   max_charge_s_id in area4: chargs_no_phot_he_gamma_a4.append(sum(qs))
                if   max_charge_s_id in area5: chargs_no_phot_he_gamma_a5.append(sum(qs))
            else:
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                if len(evt_sns) == 0:
                    continue
                ids, pos, qs, _, _, _ = pbf.info_from_the_tiles(DataSiPM_idx, evt_sns)
                if len(qs) == 0:
                    continue
                max_charge_s_id = ids[np.argmax(qs)]
                if   max_charge_s_id in area0: chargs_no_phot_no_he_gamma_a0.append(sum(qs))
                elif max_charge_s_id in area1: chargs_no_phot_no_he_gamma_a1.append(sum(qs))
                elif max_charge_s_id in area2: chargs_no_phot_no_he_gamma_a2.append(sum(qs))
                elif max_charge_s_id in area3: chargs_no_phot_no_he_gamma_a3.append(sum(qs))
                if   max_charge_s_id in area4: chargs_no_phot_no_he_gamma_a4.append(sum(qs))
                if   max_charge_s_id in area5: chargs_no_phot_no_he_gamma_a5.append(sum(qs))
        evt_ids.append(evt)

chargs_phot_he_gamma_a0 = np.array(chargs_phot_he_gamma_a0)
chargs_phot_he_gamma_a1 = np.array(chargs_phot_he_gamma_a1)
chargs_phot_he_gamma_a2 = np.array(chargs_phot_he_gamma_a2)
chargs_phot_he_gamma_a3 = np.array(chargs_phot_he_gamma_a3)
chargs_phot_he_gamma_a4 = np.array(chargs_phot_he_gamma_a4)
chargs_phot_he_gamma_a5 = np.array(chargs_phot_he_gamma_a5)

chargs_phot_no_he_gamma_a0 = np.array(chargs_phot_no_he_gamma_a0)
chargs_phot_no_he_gamma_a1 = np.array(chargs_phot_no_he_gamma_a1)
chargs_phot_no_he_gamma_a2 = np.array(chargs_phot_no_he_gamma_a2)
chargs_phot_no_he_gamma_a3 = np.array(chargs_phot_no_he_gamma_a3)
chargs_phot_no_he_gamma_a4 = np.array(chargs_phot_no_he_gamma_a4)
chargs_phot_no_he_gamma_a5 = np.array(chargs_phot_no_he_gamma_a5)

chargs_no_phot_he_gamma_a0 = np.array(chargs_no_phot_he_gamma_a0)
chargs_no_phot_he_gamma_a1 = np.array(chargs_no_phot_he_gamma_a1)
chargs_no_phot_he_gamma_a2 = np.array(chargs_no_phot_he_gamma_a2)
chargs_no_phot_he_gamma_a3 = np.array(chargs_no_phot_he_gamma_a3)
chargs_no_phot_he_gamma_a4 = np.array(chargs_no_phot_he_gamma_a4)
chargs_no_phot_he_gamma_a5 = np.array(chargs_no_phot_he_gamma_a5)

chargs_no_phot_no_he_gamma_a0 = np.array(chargs_no_phot_no_he_gamma_a0)
chargs_no_phot_no_he_gamma_a1 = np.array(chargs_no_phot_no_he_gamma_a1)
chargs_no_phot_no_he_gamma_a2 = np.array(chargs_no_phot_no_he_gamma_a2)
chargs_no_phot_no_he_gamma_a3 = np.array(chargs_no_phot_no_he_gamma_a3)
chargs_no_phot_no_he_gamma_a4 = np.array(chargs_no_phot_no_he_gamma_a4)
chargs_no_phot_no_he_gamma_a5 = np.array(chargs_no_phot_no_he_gamma_a5)

evt_ids   = np.array(evt_ids)

np.savez(evt_file, chargs_phot_he_gamma_a0=chargs_phot_he_gamma_a0, chargs_phot_he_gamma_a1=chargs_phot_he_gamma_a1, chargs_phot_he_gamma_a2=chargs_phot_he_gamma_a2,
        chargs_phot_he_gamma_a3=chargs_phot_he_gamma_a3, chargs_phot_he_gamma_a4=chargs_phot_he_gamma_a4, chargs_phot_he_gamma_a5=chargs_phot_he_gamma_a5,
        chargs_phot_no_he_gamma_a0=chargs_phot_no_he_gamma_a0, chargs_phot_no_he_gamma_a1=chargs_phot_no_he_gamma_a1, chargs_phot_no_he_gamma_a2=chargs_phot_no_he_gamma_a2,
        chargs_phot_no_he_gamma_a3=chargs_phot_no_he_gamma_a3, chargs_phot_no_he_gamma_a4=chargs_phot_no_he_gamma_a4, chargs_phot_no_he_gamma_a5=chargs_phot_no_he_gamma_a5,
        chargs_no_phot_he_gamma_a0=chargs_no_phot_he_gamma_a0, chargs_no_phot_he_gamma_a1=chargs_no_phot_he_gamma_a1, chargs_no_phot_he_gamma_a2=chargs_no_phot_he_gamma_a2,
        chargs_no_phot_he_gamma_a3=chargs_no_phot_he_gamma_a3, chargs_no_phot_he_gamma_a4=chargs_no_phot_he_gamma_a4, chargs_no_phot_he_gamma_a5=chargs_no_phot_he_gamma_a5,
        chargs_no_phot_no_he_gamma_a0=chargs_no_phot_no_he_gamma_a0, chargs_no_phot_no_he_gamma_a1=chargs_no_phot_no_he_gamma_a1, chargs_no_phot_no_he_gamma_a2=chargs_no_phot_no_he_gamma_a2,
        chargs_no_phot_no_he_gamma_a3=chargs_no_phot_no_he_gamma_a3, chargs_no_phot_no_he_gamma_a4=chargs_no_phot_no_he_gamma_a4, chargs_no_phot_no_he_gamma_a5=chargs_no_phot_no_he_gamma_a5,
        evt_ids=evt_ids)

print(datetime.datetime.now())
