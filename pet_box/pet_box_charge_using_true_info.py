import sys
import argparse
import datetime
import tables   as tb
import numpy    as np
import pandas   as pd

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

""" To run thie script
python pet_box_charge.py 2500 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
"""

print(datetime.datetime.now())

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('thr_ch_start' , type = int, help = "init threshold in charge"  )
    parser.add_argument('thr_ch_nsteps', type = int, help = "numb steps thrs in charge" )
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()

def select_phot_pet_box(evt_parts, evt_hits):
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]
    ids          = sel_vol_name.particle_id.values

    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies = energies.reset_index()

    sel1 = rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)
    sel2 = rf.lower_or_equal  (energies.energy, 0.48,     allowed_error=1.e-6)
    energy_sel = energies[sel1 & sel2]

    sel_vol_name_e  = sel_vol_name[sel_vol_name.particle_id.isin(energy_sel.particle_id)]

    primaries = evt_parts[evt_parts.primary == True]
    sel_all   = sel_vol_name_e[sel_vol_name_e.mother_id.isin(primaries.particle_id.values)]
    if len(sel_all) == 0:
        return False
    else:
        return True

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
thr_ch_start  = arguments.thr_ch_start
thr_ch_nsteps = arguments.thr_ch_nsteps
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

evt_file   = f'{out_path}/pet_box_charge_true_info_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

tot_charges = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
evt_ids     = []

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
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]
        if select_phot_pet_box(evt_parts, evt_hits):
            for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=n_th)
                if len(evt_sns) == 0:
                    continue

                sipms        = DataSiPM_idx.loc[evt_sns.sensor_id]
                sns_ids      = sipms.index.astype('int64').values
                sns_pos      = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
                sns_charges  = evt_sns.charge
                sel          = sipms.Z.values<0 #Plane with high number of sensors
                sns, pos, qs = sns_ids[sel], sns_pos[sel], sns_charges[sel]
                tot_charges[n_th].append(sum(qs))
            evt_ids.append(evt)
        else:
            continue

tot_charges = np.array(tot_charges)
evt_ids     = np.array(evt_ids)

np.savez(evt_file, tot_charges=tot_charges, evt_ids=evt_ids)

print(datetime.datetime.now())
