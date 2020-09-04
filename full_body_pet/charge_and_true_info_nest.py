import sys
import argparse
import datetime
import numpy    as np
import pandas   as pd

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcparticles

"""
python charge_and_true_info.py 22 1 /Users/carmenromoluque/Desktop/ full_body_iradius380mm_z200cm_depth4cm_pitch7mm /Users/carmenromoluque/Desktop/
"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize")
    parser.add_argument('events_path',             help = "input files path"          )
    parser.add_argument('file_name'  ,             help = "name of input files"       )
    parser.add_argument('data_path'  ,             help = "output files path"         )
    return parser.parse_args()


def select_photoelectric(evt_parts, evt_hits):
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]
    ids          = sel_vol_name.particle_id.values

    sel_hits   = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies   = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies   = energies.reset_index()
    energy_sel = energies[rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]

    sel_vol_name_e  = sel_vol_name[sel_vol_name.particle_id.isin(energy_sel.particle_id)]

    primaries = evt_parts[evt_parts.primary == True]
    sel_all   = sel_vol_name_e[sel_vol_name_e.mother_id.isin(primaries.particle_id.values)]
    if len(sel_all) == 0:
        return (False, [])

    ### Once the event has passed the selection, let's calculate the true position(s)
    ids      = sel_all.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)

    sel_hits = sel_hits.groupby(['particle_id'])
    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    return (True, true_pos)

arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path

evt_ids         = []
tot_charges     = []
threshold       = 2
num_coinc       = 0
num_singl_phots = 0

evt_file = f"{data_path}/full_body_charge_and_true_info_nest_{start}_{numb}_{threshold}"

for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number}.pet.h5"
    try:
        sns_response = pd.read_hdf(filename, 'MC/waveforms')
        sens_pos     = pd.read_hdf(filename, 'MC/sensor_positions')
    except ValueError:
        print(f'File {filename} not found')
        continue
    except OSError:
        print(f'File {filename} not found')
        continue
    except KeyError:
        print(f'No object named MC/waveforms in file {filename}')
        sns_response = pd.read_hdf(filename, 'MC/sns_response')
        sens_pos     = pd.read_hdf(filename, 'MC/sns_positions')

    print(f'Analyzing file {filename}')

    particles    = load_mcparticles(filename)
    hits         = load_mchits     (filename)

    DataSiPM     = sens_pos.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    sel_df = rf.find_SiPMs_over_threshold(sns_response, threshold=threshold)
    events = particles.event_id.unique()

    for evt in events:
        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        evt_sns   = sel_df   [sel_df   .event_id == evt]

        #select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits, every_single=True)
        select, true_pos = select_photoelectric(evt_parts, evt_hits)

        if len(true_pos) == 1:
            num_singl_phots += 1
        elif len(true_pos) == 2:
            num_coinc += 1
        else:
            continue

        max_sns = evt_sns[evt_sns.charge == evt_sns.charge.max()]
        ## If by chance two sensors have the maximum charge, choose one (arbitrarily)
        if len(max_sns != 1):
            max_sns = max_sns[max_sns.sensor_id == max_sns.sensor_id.min()]
        max_sipm = DataSiPM_idx.loc[max_sns.sensor_id]
        max_pos  = np.array([max_sipm.X.values, max_sipm.Y.values, max_sipm.Z.values]).transpose()[0]

        sipms         = DataSiPM_idx.loc[evt_sns.sensor_id]
        sns_ids       = sipms.index.astype('int64').values
        sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
        sns_charges   = evt_sns.charge

        sns1, sns2, pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_ids, sns_positions,
                                                                            sns_charges, max_pos)
        tot_q1 = sum(q1)
        tot_q2 = sum(q2)

        tot_charges.append(tot_q1)
        tot_charges.append(tot_q2)
        evt_ids.append(evt)

np.savez(evt_file, evt_ids=evt_ids, num_coinc=num_coinc, num_singl_phots=num_singl_phots, tot_charges=np.array(tot_charges))

print(datetime.datetime.now())
