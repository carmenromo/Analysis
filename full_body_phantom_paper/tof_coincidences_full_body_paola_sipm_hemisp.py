import sys
import datetime
import argparse
import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.core         import system_of_units as units

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import antea.elec.tof_functions as tf
import antea.mcsim.sensor_functions as snsf

from antea.utils.map_functions import load_map


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('rpos_file'    ,             help = "Rpos table"                )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()


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


print(datetime.datetime.now())

arguments = parse_args(sys.argv)
start     = arguments.first_file
numb      = arguments.n_files
in_path   = arguments.in_path
file_name = arguments.file_name
rpos_file = arguments.rpos_file
out_path  = arguments.out_path

thr_r   = 4
thr_phi = 4
thr_z   = 4
thr_e   = 2

evt_file = out_path + f'full_body_phantom_paper_coinc2_{start}_{numb}'
print(f'Using r map: {rpos_file}')

Rpos = load_map(rpos_file,
                 group  = "Radius",
                 node   = f"f{int(thr_r)}pes150bins",
                 x_name = "PhiRms",
                 y_name = "Rpos",
                 u_name = "RposUncertainty")

charge_range = (2000, 3000) # pde 0.30, n=1.6

print(f'Charge range = {charge_range}')
c0 = c1 = c2 = c3 = c4 = 0
bad = 0

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

sns_response1, sns_response2    = [], []
max_hit_distance1, max_hit_distance2 = [], []
event_ids = []
max_pos  = []


for ifile in range(start, start+numb):
    filename = in_path + file_name + f'.{ifile}.h5'
    try:
        sns_response = pd.read_hdf(filename, 'MC/sns_response')
    except ValueError:
        print('File {} not found'.format(filename))
        continue
    except OSError:
        print('File {} not found'.format(filename))
        continue
    except KeyError:
        print('No object named MC/sns_response in file {0}'.format(filename))
        continue
    print('Analyzing file {0}'.format(filename))

    particles = pd.read_hdf(filename, 'MC/particles')
    hits      = pd.read_hdf(filename, 'MC/hits')
    sns_response = snsf.apply_charge_fluctuation(sns_response, DataSiPM_idx)

    events = particles.event_id.unique()

    for evt in events:

        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=thr_e)
        if len(evt_sns) == 0:
            continue

        ids_over_thr = evt_sns.sensor_id.astype('int64').values

        evt_parts = particles[particles.event_id       == evt]
        evt_hits  = hits[hits.event_id                 == evt]

        pos1, pos2, q1, q2, true_pos1, true_pos2, true_t1, true_t2, sns1, sns2, max_p = reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)
        if len(pos1) == 0 or len(pos2) == 0:
            c0 += 1
            continue

        q1   = np.array(q1);
        q2   = np.array(q2);
        pos1 = np.array(pos1);
        pos2 = np.array(pos2);

        ## Calculate R
        r1 = r2 = None

        sel1_r = q1>thr_r
        q1r    = q1[sel1_r]
        pos1r  = pos1[sel1_r]
        sel2_r = q2>thr_r
        q2r    = q2[sel2_r]
        pos2r  = pos2[sel2_r]
        if len(pos1r) == 0 or len(pos2r) == 0:
            c1 += 1
            continue

        pos1_phi = rf.from_cartesian_to_cyl(np.array(pos1r))[:,1]
        diff_sign = min(pos1_phi ) < 0 < max(pos1_phi)
        if diff_sign & (np.abs(np.min(pos1_phi))>np.pi/2.):
            pos1_phi[pos1_phi<0] = np.pi + np.pi + pos1_phi[pos1_phi<0]
        mean_phi = np.average(pos1_phi, weights=q1r)
        var_phi1 = np.average((pos1_phi-mean_phi)**2, weights=q1r)
        r1  = Rpos(np.sqrt(var_phi1)).value

        pos2_phi = rf.from_cartesian_to_cyl(np.array(pos2r))[:,1]
        diff_sign = min(pos2_phi ) < 0 < max(pos2_phi)
        if diff_sign & (np.abs(np.min(pos2_phi))>np.pi/2.):
            pos2_phi[pos2_phi<0] = np.pi + np.pi + pos2_phi[pos2_phi<0]
        mean_phi = np.average(pos2_phi, weights=q2r)
        var_phi2 = np.average((pos2_phi-mean_phi)**2, weights=q2r)
        r2  = Rpos(np.sqrt(var_phi2)).value

        sel1_phi = q1>thr_phi
        q1phi    = q1[sel1_phi]
        pos1phi  = pos1[sel1_phi]
        sel2_phi = q2>thr_phi
        q2phi    = q2[sel2_phi]
        pos2phi  = pos2[sel2_phi]
        if len(q1phi) == 0 or len(q2phi) == 0:
            c2 += 1
            continue

        phi1 = phi2 = None
        reco_cart_pos = np.average(pos1phi, weights=q1phi, axis=0)
        phi1 = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])
        reco_cart_pos = np.average(pos2phi, weights=q2phi, axis=0)
        phi2 = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])


        sel1_z = q1>thr_z
        q1z    = q1[sel1_z]
        pos1z  = pos1[sel1_z]
        sel2_z = q2>thr_z
        q2z    = q2[sel2_z]
        pos2z  = pos2[sel2_z]
        if len(q1z) == 0 or len(q2z) == 0:
            c3 += 1
            continue

        z1 = z2 = None
        reco_cart_pos = np.average(pos1z, weights=q1z, axis=0)
        z1 = reco_cart_pos[2]
        reco_cart_pos = np.average(pos2z, weights=q2z, axis=0)
        z2 = reco_cart_pos[2]

        sel1_e = q1>thr_e
        q1e    = q1[sel1_e]
        sel2_e = q2>thr_e
        q2e    = q2[sel2_e]
        if len(q1e) == 0 or len(q2e) == 0:
            c4 += 1
            continue


        ## extract information about the interaction being photoelectric-like
        positions         = np.array([evt_hits.x, evt_hits.y, evt_hits.z]).transpose()
        scalar_products1 = positions.dot(true_pos1)
        hits1 = evt_hits[scalar_products1 >= 0]
        pos_hits1  = np.array([hits1.x, hits1.y, hits1.z]).transpose()
        distances1 = np.linalg.norm(np.subtract(pos_hits1, true_pos1), axis=1)
        max_dist1  = distances1.max()

        hits2 = evt_hits[scalar_products1 < 0]
        pos_hits2  = np.array([hits2.x, hits2.y, hits2.z]).transpose()
        distances2 = np.linalg.norm(np.subtract(pos_hits2, true_pos2), axis=1)
        max_dist2  = distances2.max()

        event_ids        .append(evt)
        reco_r1          .append(r1)
        reco_phi1        .append(phi1)
        reco_z1          .append(z1)
        true_r1          .append(np.sqrt(true_pos1[0]**2 + true_pos1[1]**2))
        true_phi1        .append(np.arctan2(true_pos1[1], true_pos1[0]))
        true_z1          .append(true_pos1[2])
        sns_response1    .append(sum(q1e))
        max_hit_distance1.append(max_dist1)
        reco_r2          .append(r2)
        reco_phi2        .append(phi2)
        reco_z2          .append(z2)
        true_r2          .append(np.sqrt(true_pos2[0]**2 + true_pos2[1]**2))
        true_phi2        .append(np.arctan2(true_pos2[1], true_pos2[0]))
        true_z2          .append(true_pos2[2])
        sns_response2    .append(sum(q2e))
        max_hit_distance2.append(max_dist2)
        max_pos          .append(max_p)


a_true_r1           = np.array(true_r1)
a_true_phi1         = np.array(true_phi1)
a_true_z1           = np.array(true_z1)
a_reco_r1           = np.array(reco_r1)
a_reco_phi1         = np.array(reco_phi1)
a_reco_z1           = np.array(reco_z1)
a_sns_response1     = np.array(sns_response1)
a_max_hit_distance1 = np.array(max_hit_distance1)
max_pos             = np.array(max_pos)

a_true_r2           = np.array(true_r2)
a_true_phi2         = np.array(true_phi2)
a_true_z2           = np.array(true_z2)
a_reco_r2           = np.array(reco_r2)
a_reco_phi2         = np.array(reco_phi2)
a_reco_z2           = np.array(reco_z2)
a_sns_response2     = np.array(sns_response2)
a_max_hit_distance2 = np.array(max_hit_distance2)

a_event_ids = np.array(event_ids)

np.savez(evt_file,
         a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1,
         a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
         a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
         a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
         a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2,
         a_max_hit_distance1=a_max_hit_distance1, a_max_hit_distance2=a_max_hit_distance2,
         a_event_ids=a_event_ids, max_pos=max_pos)

print('Not a coincidence: {}'.format(c0))
print(f'Number of coincidences: {len(a_event_ids)}')
print('Not passing threshold r = {}, phi = {}, z = {}, E = {}'.format(c1, c2, c3, c4))
