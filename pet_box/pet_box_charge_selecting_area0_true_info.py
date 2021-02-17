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
python pet_box_charge_selecting_area0_true_info.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

evt_file   = f'{out_path}/pet_box_charge_select_areas_true_info_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

# phot + he_gamma
chargs_phot_he_gamma_a0 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_he_gamma_a1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_he_gamma_a2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_he_gamma_a3 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_he_gamma_a4 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_he_gamma_a5 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

# phot + NO he_gamma
chargs_phot_no_he_gamma_a0 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_no_he_gamma_a1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_no_he_gamma_a2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_no_he_gamma_a3 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_no_he_gamma_a4 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_phot_no_he_gamma_a5 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

# NO phot + he_gamma
chargs_no_phot_he_gamma_a0 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_he_gamma_a1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_he_gamma_a2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_he_gamma_a3 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_he_gamma_a4 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_he_gamma_a5 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

# NO phot + NO he_gamma
chargs_no_phot_no_he_gamma_a0 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_no_he_gamma_a1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_no_he_gamma_a2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_no_he_gamma_a3 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_no_he_gamma_a4 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
chargs_no_phot_no_he_gamma_a5 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evt_ids   = []

area0 = [8, 28, 37, 57]
area1 = [7, 15, 16, 19, 20, 27, 38, 45, 46, 49, 50, 58]
area2 = [6, 10, 11, 12, 14, 18, 22, 23, 24, 26, 39, 41, 42, 43, 47, 51, 53, 54, 55, 59]
area3 = [1, 2, 3, 4, 5, 9, 13, 17, 21, 25, 29, 30, 31, 32, 33, 34, 35, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63, 64]
area4 = area0 + area1
area5 = area0 + area1 + area2

def select_gamma_high_energy(evt_parts, evt_hits):
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]

    gamma_he  = evt_parts[evt_parts.kin_energy == 1.274537]
    sel_mother = sel_vol_name[sel_vol_name.mother_id.isin(gamma_he.particle_id.values)]

    ids      = sel_mother.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies = energies.reset_index()

    energy_sel = energies[energies.energy == energies.energy.max()]
    sel_all = sel_mother[sel_mother.particle_id.isin(energy_sel.particle_id)]

    if len(sel_all)==0:
        return (False, [])

    ### Once the event has passed the selection, let's calculate the true position(s)
    ids      = sel_all.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)

    sel_hits = sel_hits.groupby(['particle_id'])
    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    return (True, np.array(true_pos))

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

        he_gamma, true_pos_he = select_gamma_high_energy(evt_parts, evt_hits)
        phot, true_pos_phot   = mcf.select_photoelectric(evt_parts, evt_hits)

        sel_neg_phot = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_neg_phot[sel_neg_phot<0]

        sel_neg_he = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_neg_he[sel_neg_he<0]

        if phot and len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
            if he_gamma and len(sel_neg_he)>0: ### Be careful with the meaning of this condition
                for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                    evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                    if len(evt_sns) == 0:
                        continue
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    if len(qs) == 0:
                        continue
                    max_charge_s_id = ids[np.argmax(qs)]
                    if   max_charge_s_id in area0: chargs_phot_he_gamma_a0[n_th].append(sum(qs))
                    elif max_charge_s_id in area1: chargs_phot_he_gamma_a1[n_th].append(sum(qs))
                    elif max_charge_s_id in area2: chargs_phot_he_gamma_a2[n_th].append(sum(qs))
                    elif max_charge_s_id in area3: chargs_phot_he_gamma_a3[n_th].append(sum(qs))
                    if   max_charge_s_id in area4: chargs_phot_he_gamma_a4[n_th].append(sum(qs))
                    if   max_charge_s_id in area5: chargs_phot_he_gamma_a5[n_th].append(sum(qs))
            else:
                for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                    evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                    if len(evt_sns) == 0:
                        continue
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    if len(qs) == 0:
                        continue
                    max_charge_s_id = ids[np.argmax(qs)]
                    if   max_charge_s_id in area0: chargs_phot_no_he_gamma_a0[n_th].append(sum(qs))
                    elif max_charge_s_id in area1: chargs_phot_no_he_gamma_a1[n_th].append(sum(qs))
                    elif max_charge_s_id in area2: chargs_phot_no_he_gamma_a2[n_th].append(sum(qs))
                    elif max_charge_s_id in area3: chargs_phot_no_he_gamma_a3[n_th].append(sum(qs))
                    if   max_charge_s_id in area4: chargs_phot_no_he_gamma_a4[n_th].append(sum(qs))
                    if   max_charge_s_id in area5: chargs_phot_no_he_gamma_a5[n_th].append(sum(qs))
        else:
            if he_gamma and len(sel_neg_he)>0: ### Be careful with the meaning of this condition
                for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                    evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                    if len(evt_sns) == 0:
                        continue
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    if len(qs) == 0:
                        continue
                    max_charge_s_id = ids[np.argmax(qs)]
                    if   max_charge_s_id in area0: chargs_no_phot_he_gamma_a0[n_th].append(sum(qs))
                    elif max_charge_s_id in area1: chargs_no_phot_he_gamma_a1[n_th].append(sum(qs))
                    elif max_charge_s_id in area2: chargs_no_phot_he_gamma_a2[n_th].append(sum(qs))
                    elif max_charge_s_id in area3: chargs_no_phot_he_gamma_a3[n_th].append(sum(qs))
                    if   max_charge_s_id in area4: chargs_no_phot_he_gamma_a4[n_th].append(sum(qs))
                    if   max_charge_s_id in area5: chargs_no_phot_he_gamma_a5[n_th].append(sum(qs))
            else:
                for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                    evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
                    if len(evt_sns) == 0:
                        continue
                    ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                    if len(qs) == 0:
                        continue
                    max_charge_s_id = ids[np.argmax(qs)]
                    if   max_charge_s_id in area0: chargs_no_phot_no_he_gamma_a0[n_th].append(sum(qs))
                    elif max_charge_s_id in area1: chargs_no_phot_no_he_gamma_a1[n_th].append(sum(qs))
                    elif max_charge_s_id in area2: chargs_no_phot_no_he_gamma_a2[n_th].append(sum(qs))
                    elif max_charge_s_id in area3: chargs_no_phot_no_he_gamma_a3[n_th].append(sum(qs))
                    if   max_charge_s_id in area4: chargs_no_phot_no_he_gamma_a4[n_th].append(sum(qs))
                    if   max_charge_s_id in area5: chargs_no_phot_no_he_gamma_a5[n_th].append(sum(qs))
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
