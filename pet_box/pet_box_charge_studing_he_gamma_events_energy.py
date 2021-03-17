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
python pet_box_charge_studing_he_gamma_events_energy.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
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

evt_file  = f'{out_path}/pet_box_charge_studying_he_gamma_and_phot_evts_energy_{start}_{numb}'

energy_he_gamma_pos  = []
energy_he_gamma_neg  = []
energy_phot1         = []
energy_phot2         = []
charges_he_gamma_pos = []
charges_he_gamma_neg = []
charges_phot1        = []
charges_phot2        = []
evts_he_gamma_pos    = []
evts_he_gamma_neg    = []
evts_phot1           = []
evts_phot2           = []

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

        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
        if len(evt_sns) == 0:
            continue

        sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
        sel_name     =  evt_parts.particle_name == 'e-'
        sel_vol_name = evt_parts[sel_volume & sel_name]

        ## High energy gamma:
        gamma_he  = evt_parts[evt_parts.kin_energy == 1.274537]
        sel_mother_g = sel_vol_name[sel_vol_name.mother_id.isin(gamma_he.particle_id.values)]

        ids_g      = sel_mother_g.particle_id.values
        sel_hits_g = mcf.find_hits_of_given_particles(ids_g, evt_hits)
        energies_g = sel_hits_g.groupby(['particle_id'])[['energy']].sum()
        energies_g = energies_g.reset_index()

        energy_sel_g = energies_g[energies_g.energy == energies_g.energy.max()]
        if len(energy_sel_g)!=0:
            sel_all_g = sel_mother_g[sel_mother_g.particle_id.isin(energy_sel_g.particle_id)]
            if sel_all_g.initial_momentum_z.values[0]<0:
                ids_g_neg, pos_g_neg, qs_g_neg = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                energy_he_gamma_neg .append(energy_sel_g.energy.values[0])
                charges_he_gamma_neg.append(sum(qs_g_neg))
                evts_he_gamma_neg   .append(evt)
            else:
                ids_g_pos, pos_g_pos, qs_g_pos = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
                energy_he_gamma_pos .append(energy_sel_g.energy.values[0])
                charges_he_gamma_pos.append(sum(qs_g_pos))
                evts_he_gamma_pos   .append(evt)

        ## 511keV gammas:
        phot1         = evt_parts[(evt_parts.kin_energy == 0.510999) & (evt_parts.initial_momentum_z < 0)]
        phot2         = evt_parts[(evt_parts.kin_energy == 0.510999) & (evt_parts.initial_momentum_z > 0)]
        sel_mother_p1 = sel_vol_name[sel_vol_name.mother_id.isin(phot1.particle_id.values)]
        sel_mother_p2 = sel_vol_name[sel_vol_name.mother_id.isin(phot2.particle_id.values)]

        ids_p1      = sel_mother_p1.particle_id.values
        sel_hits_p1 = mcf.find_hits_of_given_particles(ids_p1, evt_hits)
        energies_p1 = sel_hits_p1.groupby(['particle_id'])[['energy']].sum()
        energies_p1 = energies_p1.reset_index()

        energy_sel_p1 = energies_p1[energies_p1.energy == energies_p1.energy.max()]
        if len(energy_sel_p1)!=0:
            ids_p1, pos_p1, qs_p1 = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
            energy_phot1 .append(energy_sel_p1.energy.values[0])
            charges_phot1.append(sum(qs_p1))
            evts_phot1   .append(evt)


        ids_p2      = sel_mother_p2.particle_id.values
        sel_hits_p2 = mcf.find_hits_of_given_particles(ids_p2, evt_hits)
        energies_p2 = sel_hits_p2.groupby(['particle_id'])[['energy']].sum()
        energies_p2 = energies_p2.reset_index()

        energy_sel_p2 = energies_p2[energies_p2.energy == energies_p2.energy.max()]
        if len(energy_sel_p2)!=0:
            ids_p2, pos_p2, qs_p2 = pbf.info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns)
            energy_phot2 .append(energy_sel_p2.energy.values[0])
            charges_phot2.append(sum(qs_p2))
            evts_phot2   .append(evt)

energy_he_gamma_pos  = np.array(energy_he_gamma_pos)
energy_he_gamma_neg  = np.array(energy_he_gamma_neg)
energy_phot1         = np.array(energy_phot1)
energy_phot2         = np.array(energy_phot2)
charges_he_gamma_pos = np.array(charges_he_gamma_pos)
charges_he_gamma_neg = np.array(charges_he_gamma_neg)
charges_phot1        = np.array(charges_phot1)
charges_phot2        = np.array(charges_phot2)
evts_he_gamma_pos    = np.array(evts_he_gamma_pos)
evts_he_gamma_neg    = np.array(evts_he_gamma_neg)
evts_phot1           = np.array(evts_phot1)
evts_phot2           = np.array(evts_phot2)

np.savez(evt_file, energy_he_gamma_pos=energy_he_gamma_pos, energy_he_gamma_neg=energy_he_gamma_neg, energy_phot1=energy_phot1, energy_phot2=energy_phot2,
         charges_he_gamma_pos=charges_he_gamma_pos, charges_he_gamma_neg=charges_he_gamma_neg, charges_phot1=charges_phot1, charges_phot2=charges_phot2,
         evts_he_gamma_pos=evts_he_gamma_pos, evts_he_gamma_neg=evts_he_gamma_neg, evts_phot1=evts_phot1, evts_phot2=evts_phot2)

print(datetime.datetime.now())
