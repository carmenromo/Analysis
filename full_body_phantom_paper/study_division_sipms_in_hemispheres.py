
import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import antea.io      .mc_io        as mcio
import antea.database.load_db      as db
import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()

print(datetime.datetime.now())

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

def divide_sipms_in_two_hemispheres(sns_ids, sns_positions, sns_charges, reference_pos):
    q1,   q2   = [], []
    pos1, pos2 = [], []
    id1, id2   = [], []
    for sns_id, sns_pos, charge in zip(sns_ids, sns_positions, sns_charges):
        scalar_prod = sns_pos[:2].dot(reference_pos[:2])
        if scalar_prod > 0.:
            q1  .append(charge)
            pos1.append(sns_pos)
            id1 .append(sns_id)
        else:
            q2  .append(charge)
            pos2.append(sns_pos)
            id2 .append(sns_id)

    return np.array(id1), np.array(id2), np.array(pos1), np.array(pos2), np.array(q1), np.array(q2)


def reconstruct_coincidences(sns_response, charge_range, DataSiPM_idx, particles, hits):
    if 'SensorID' in DataSiPM_idx.columns:
        DataSiPM_idx = DataSiPM_idx.set_index('SensorID')

    max_sns = sns_response[sns_response.charge == sns_response.charge.max()]
    ## If by chance two sensors have the maximum charge, choose one (arbitrarily)

    if len(max_sns != 1):
        max_sns = max_sns[max_sns.sensor_id == max_sns.sensor_id.min()]
    max_sipm = DataSiPM_idx.loc[max_sns.sensor_id]
    max_pos  = np.array([max_sipm.X.values, max_sipm.Y.values, max_sipm.Z.values]).transpose()[0]

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.astype('int64').values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = sns_response.charge

    sns1, sns2, pos1, pos2, q1, q2 = divide_sipms_in_two_hemispheres(sns_ids, sns_positions, sns_charges, max_pos)

    tot_q1 = sum(q1)
    tot_q2 = sum(q2)

    sel1 = (tot_q1 > charge_range[0]) & (tot_q1 < charge_range[1])
    sel2 = (tot_q2 > charge_range[0]) & (tot_q2 < charge_range[1])
    if not sel1 or not sel2:
        return [], [], [], [], None, None, None, None, [], [], []

    true_pos1, true_pos2, true_t1, true_t2, _, _ = rf.find_first_interactions_in_active(particles, hits)

    if not len(true_pos1) or not len(true_pos2):
        print("Cannot find two true gamma interactions for this event")
        return [], [], [], [], None, None, None, None, [], [], []

    scalar_prod = true_pos1[:2].dot(max_pos[:2])
    if scalar_prod > 0:
        int_pos1 = pos1
        int_pos2 = pos2
        int_q1   = q1
        int_q2   = q2
        int_sns1 = sns1
        int_sns2 = sns2
    else:
        int_pos1 = pos2
        int_pos2 = pos1
        int_q1   = q2
        int_q2   = q1
        int_sns1 = sns2
        int_sns2 = sns1

    return int_pos1, int_pos2, int_q1, int_q2, true_pos1, true_pos2, true_t1, true_t2, int_sns1, int_sns2, max_pos


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
DataSiPM_idx = DataSiPM.set_index('SensorID')

evt_file  = out_path + f'study_division_sipms_in_hemispheres_{start}_{numb}'

max_hit_distance1 = []
max_hit_distance2 = []
event_ids         = []


for number in range(start, start+numb):
    filename = in_path + file_name + f'.{number}.h5'
    try:
        sns_response = mcio.load_mcsns_response(filename)
    except OSError:
        print(f'File {filename} does not exist')
        continue
    print(f'file {number}')

    print(filename)
    particles    = mcio.load_mcparticles(filename)
    hits         = mcio.load_mchits(filename)

    events = particles.event_id.unique()
    charge_range = (0, 6000) # range to select photopeak - to be adjusted to the specific case

    for evt in events:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=2)
        if len(evt_sns) == 0:
            continue

        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]

        pos11, pos21, q11, q21, true_pos11, true_pos21, true_t11, true_t21, sns11, sns21, max_p =    reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)
        pos12, pos22, q12, q22, true_pos12, true_pos22, true_t12, true_t22, sns12, sns22        = rf.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)

        if len(pos11) == 0 or len(pos21) == 0 or len(pos12) == 0 or len(pos22) == 0:
            continue

        ## extract information about the interaction being photoelectric-like
        distances1 = rf.find_hit_distances_from_true_pos(evt_hits, true_pos11)
        max_dist1  = distances1.max()
        distances2 = rf.find_hit_distances_from_true_pos(evt_hits, true_pos22)
        max_dist2  = distances2.max()

        if len(sns11) != len(sns12):
            event_ids        .append(evt)
            max_hit_distance1.append(max_dist1)
            max_hit_distance2.append(max_dist2)
        #if len(sns21) != len(sns22):


a_max_hit_distance1 = np.array(max_hit_distance1)
a_max_hit_distance2 = np.array(max_hit_distance2)
a_event_ids = np.array(event_ids)

np.savez(evt_file, a_max_hit_distance1=a_max_hit_distance1, a_max_hit_distance2=a_max_hit_distance2, a_event_ids=a_event_ids)
