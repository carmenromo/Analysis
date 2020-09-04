import sys
import argparse
import datetime
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mcsns_response

"""
python statistics_of_the_events.py 22 1 /Users/carmenromoluque/Desktop/ full_body_iradius380mm_z200cm_depth4cm_pitch7mm /Users/carmenromoluque/Desktop/
"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"     )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize" )
    parser.add_argument('events_path',             help = "input files path"           )
    parser.add_argument('file_name'  ,             help = "name of input files"        )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()

def charge_selection(sns_response, DataSiPM_idx, charge_range):
    max_sns = sns_response[sns_response.charge == sns_response.charge.max()]
    if len(max_sns != 1):
        max_sns = max_sns[max_sns.sensor_id == max_sns.sensor_id.min()]
    max_sipm = DataSiPM_idx.loc[max_sns.sensor_id]
    max_pos  = np.array([max_sipm.X.values, max_sipm.Y.values, max_sipm.Z.values]).transpose()[0]

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.astype('int64').values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = sns_response.charge

    sns1, sns2, pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_ids, sns_positions,
                                                                        sns_charges, max_pos)
    tot_q1 = sum(q1)
    tot_q2 = sum(q2)

    sel1 = (tot_q1 > charge_range[0]) & (tot_q1 < charge_range[1])
    sel2 = (tot_q2 > charge_range[0]) & (tot_q2 < charge_range[1])

    if not sel1 or not sel2:
        return None, None, [], [], [], [], [], []
    else:
        return sel1, sel2, sns1, sns2, pos1, pos2, q1, q2


def select_photoelectric(evt_parts, evt_hits):
    """
    Select only the events where one or two photoelectric events occur, and nothing else.
    """
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.name == 'e-'
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
 
 
def compton_selection2(particles, hits):
    sel_volume = (particles.initial_volume == 'ACTIVE') & (particles.creator_proc == 'compt')
    sel_name   =  particles.name == 'e-'
    sel_all    = particles[sel_volume & sel_name]
    primaries  = particles[particles.primary == True]
    sel_all    = sel_all[sel_all.mother_id.isin(primaries.particle_id.values)]

    ids      = sel_all.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, hits)
    sel_hits = sel_hits.groupby(['particle_id'])
    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        hit_time      = df.time.values[0]
        cuatrivect    = np.array([np.average(hit_positions, axis=0, weights=df.energy), hit_time])
        true_pos.append(cuatrivect)

    if len(sel_all)==0:
        return False, False, [], 0, [], 0
    else:
        p1, p2 = [], []
        t1, t2 = [], []
        for p in true_pos:
            print(p)
            if true_pos[0][0].dot(p[0])>0:
                p1.append(p[0])
                t1.append(p[1])
            else:
                p2.append(p[0])
                t2.append(p[1])
        if p1:
            min_t1           = min(t1)
            pos_min_compton1 = np.array([p1[t1.index(min_t1)]])
            dist1            = np.linalg.norm(pos_min_compton1 - np.array([np.average(np.array(p1), axis=0)]))
            print('Lets see')
            print(min_t1, pos_min_compton1, dist1)
            print(np.array(p1))
            print(np.array([np.average(np.array(p1), axis=0)]))
            print(pos_min_compton1 - np.array([np.average(np.array(p1), axis=0)]))
            if p2:
                min_t2           = min(t2)
                pos_min_compton2 = np.array([p2[t2.index(min_t2)]])
                dist2            = np.linalg.norm(pos_min_compton2 - np.array([np.average(np.array(p2), axis=0)]))
                return (True, True, np.array([np.average(np.array(p1), axis=0), t1[0]]), dist1,
                        np.array([np.average(np.array(p2), axis=0), t2[0]]), dist2)
            else:
                return True, False, np.array([np.average(np.array(p1), axis=0), t1[0]]), dist1, [], 0
        else:
            min_t2           = min(t2)
            pos_min_compton2 = np.array([p2[t2.index(min_t2)]])
            dist2            = np.linalg.norm(pos_min_compton2 - np.array([np.average(np.array(p2), axis=0)]))
            return False, True, [], 0, np.array([np.average(np.array(p2), axis=0), t2[0]]), dist2


arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path

dist_first_hit_ave = []
phot_events      = []
compt_events     = []
phot_like_events = []
evt_ids          = []
min_touched_sns  = 0
threshold        = 2
charge_range     = (1050, 1300)

evt_file = f"{data_path}/full_body_statistics_{start}_{numb}_{threshold}"

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    try:
        #sns_response = load_mcsns_response(filename)
        sns_response = pd.read_hdf(filename, 'MC/waveforms')
    except ValueError:
        print(f'File {filename} not found')
        continue
    except OSError:
        print(f'File {filename} not found')
        continue
    except KeyError:
        print(f'No object named MC/waveforms in file {filename}')
        continue
    print(f'Analyzing file {filename}')

    particles    = load_mcparticles(filename)
    hits         = load_mchits     (filename)
    sens_pos     = pd.read_hdf     (filename, 'MC/sensor_positions')
    #particles    = pd.read_hdf(filename, 'MC/particles'       )
    #hits         = pd.read_hdf(filename, 'MC/hits'            )

    DataSiPM     = sens_pos.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    sel_df = rf.find_SiPMs_over_threshold(sns_response, threshold=threshold)
    events = particles.event_id.unique()

    for evt in events:
        ### Select compton events only
        evt_parts = particles   [particles   .event_id == evt]
        evt_hits  = hits        [hits        .event_id == evt]
        evt_sns   = sns_response[sns_response.event_id == evt]

        sel_df = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
        select, true_pos = select_photoelectric(evt_parts, evt_hits)

        sel1, sel2, sns1, sns2, pos1, pos2, q1, q2 = charge_selection(sel_df, DataSiPM_idx, charge_range)
        if not sel1 and not sel2:
            continue

        compt1, compt2, true_pos1, d1, true_pos2, d2 = compton_selection2(evt_parts, evt_hits)
        print(compt1, compt2, true_pos1, d1, true_pos2, d2)

        phot_events       .append(len(true_pos))
        compt_events      .append((compt1, compt2))
        dist_first_hit_ave.append((d1, d2))
        evt_ids           .append(evt)


a_compt_events       = np.array(    compt_events)
a_dist_first_hit_ave = np.array(dist_first_hit_ave)
a_phot_events        = np.array(phot_events)
a_evt_ids            = np.array(evt_ids)

np.savez(evt_file, evt_ids=a_evt_ids, compt_events=a_compt_events, dist_first_hit_ave=a_dist_first_hit_ave, phot_events=a_phot_events)

print(datetime.datetime.now())
