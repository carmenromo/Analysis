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

evt_file  = f'{out_path}/pet_box_charge_studying_he_gamma_evts_energy_{start}_{numb}'

energy_he_gamma = []

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

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]

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
        if len(energy_sel)!=0:
            energy_he_gamma.append(energy_sel.energy.values[0])

energy_he_gamma = np.array(energy_he_gamma)

np.savez(evt_file, energy_he_gamma=energy_he_gamma)

print(datetime.datetime.now())
