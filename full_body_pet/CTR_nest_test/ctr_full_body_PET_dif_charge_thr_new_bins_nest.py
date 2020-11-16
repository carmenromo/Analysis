import sys
import argparse
import datetime
import tables   as tb
import numpy    as np
import pandas   as pd

import antea.database.load_db    as db
import antea.reco.reco_functions as rf
import antea.elec.tof_functions  as tf

from antea.utils.map_functions import load_map
from antea.io   .mc_io         import load_mchits
from antea.io   .mc_io         import load_mcparticles
from antea.io   .mc_io         import load_mcsns_response
from antea.io   .mc_io         import load_mcTOFsns_response
from antea.io   .mc_io         import read_sensor_bin_width_from_conf

from invisible_cities.core import system_of_units as units

"""
python ctr_full_body_PET_dif_charge_thr_new_bins_nest.py 0 1 2 4 4 2 /Users/carmenromoluque/Desktop/ full_body_center_nest_fixed /Users/carmenromoluque/Analysis/fastmc/r_table_full_body_195cm_thr2pes.h5 /Users/carmenromoluque/Desktop/
"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"     )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize" )
    parser.add_argument('thr_r'      , type = int, help = "threshold in r coordinate"  )
    parser.add_argument('thr_phi'    , type = int, help = "threshold in phi coordinate")
    parser.add_argument('thr_z'      , type = int, help = "threshold in z coordinate"  )
    parser.add_argument('thr_e'      , type = int, help = "threshold in the energy"    )
    parser.add_argument('events_path',             help = "input files path"           )
    parser.add_argument('file_name'  ,             help = "name of input files"        )
    parser.add_argument('rpos_file'  ,             help = "File of the Rpos"           )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()


def find_selected_times_of_sensors(evt_sns, evt_tof, sns_ids, num_sel_sns, DataSiPM_idx):
    tof              = evt_tof[evt_tof.sensor_id.isin(sns_ids)]
    min_ts           = tof.groupby(['sensor_id'])[['time_bin']].min().sort_values('time_bin')
    mean_t_sel_sns   = min_ts[:num_sel_sns].time_bin.mean()
    ids_sel_sns      = min_ts[:num_sel_sns].index.values
    evt_sns_sel_sns  = evt_sns[evt_sns.sensor_id.isin(-ids_sel_sns)]
    charges_sel_sns  = evt_sns_sel_sns.groupby(['sensor_id'])[['charge']].sum().values.T[0]
    sipms            = DataSiPM_idx.loc[evt_sns_sel_sns.sensor_id]
    sns_positions    = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).T
    weig_pos_sel_sns = np.average(                sns_positions, axis=0, weights=charges_sel_sns)
    weig_t_sel_sns   = np.average(min_ts[:num_sel_sns].time_bin, axis=0, weights=charges_sel_sns)
    return ids_sel_sns, charges_sel_sns, mean_t_sel_sns, weig_t_sel_sns, weig_pos_sel_sns


def sensor_position(s_id, sipms):
    xpos = sipms[sipms.index==s_id].X.unique()[0]
    ypos = sipms[sipms.index==s_id].Y.unique()[0]
    zpos = sipms[sipms.index==s_id].Z.unique()[0]
    return np.array([xpos, ypos, zpos])



arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
thr_r      = arguments.thr_r
thr_phi    = arguments.thr_phi
thr_z      = arguments.thr_z
thr_e      = arguments.thr_e
eventsPath = arguments.events_path
file_name  = arguments.file_name
rpos_file  = arguments.rpos_file
data_path  = arguments.data_path

evt_file  = f"{data_path}/full_body_nest_ctr_tof_new_bins_{start}_{numb}_{thr_r}_{thr_phi}_{thr_z}_{thr_e}"
Rpos = load_map(rpos_file, group="Radius",
                node=f"f{thr_r}pes200bins",
                x_name='RmsPhi',
                y_name='Rpos',
                u_name='Uncertainty')

### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

c0 = c1 = c2 = c3 = c4 = 0

### TOF elec parameters:
first_sipm     = 1000 #DataSiPM_idx.index.min()
time_window    = 10000
time_bin       = 5
tau_sipm       = [100, 15000]
time           = np.arange(0, 80000, time_bin)
time           = time+(time_bin/2)
spe_resp, norm = tf.apply_spe_dist(time, tau_sipm)

timestamp_thr   = [0, 0.5]

time_diff = [[] for k in range(len(timestamp_thr))]
pos_cart1 = []
pos_cart2 = []
event_ids = []

ave_speed_in_LXe = 0.210       # mm/ps
speed_in_vacuum  = 0.299792458 # mm/ps


for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number}.pet.h5"
    try:
        sns_response = load_mcsns_response(filename)
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

    tof_bin_size = read_sensor_bin_width_from_conf(filename, tof=True)

    sns_response_tof = load_mcTOFsns_response(filename)
    particles        = load_mcparticles(filename)
    hits             = load_mchits(filename)

    events = particles.event_id.unique()

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()

    charge_range = (1050, 1300)

    for evt in events[:]:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=2)
        if len(evt_sns) == 0:
            continue

        evt_parts = particles       [particles       .event_id == evt]
        evt_hits  = hits            [hits            .event_id == evt]
        evt_tof   = sns_response_tof[sns_response_tof.event_id == evt]

        pos1, pos2, q1, q2, _, _, _, _, _, _ = rf.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)

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
        q1r    = q1  [sel1_r]
        pos1r  = pos1[sel1_r]
        sel2_r = q2>thr_r
        q2r    = q2  [sel2_r]
        pos2r  = pos2[sel2_r]

        if len(pos1r) == 0 or len(pos2r) == 0:
            c1 += 1
            continue

        pos1_phi  = rf.from_cartesian_to_cyl(np.array(pos1r))[:,1]
        diff_sign = min(pos1_phi) < 0 < max(pos1_phi)
        if diff_sign & (np.abs(np.min(pos1_phi))>np.pi/2.):
            pos1_phi[pos1_phi<0] = np.pi + np.pi + pos1_phi[pos1_phi<0]
        mean_phi = np.average(pos1_phi, weights=q1r)
        var_phi1 = np.average((pos1_phi-mean_phi)**2, weights=q1r)
        r1  = Rpos(np.sqrt(var_phi1)).value

        pos2_phi  = rf.from_cartesian_to_cyl(np.array(pos2r))[:,1]
        diff_sign = min(pos2_phi ) < 0 < max(pos2_phi)
        if diff_sign & (np.abs(np.min(pos2_phi))>np.pi/2.):
            pos2_phi[pos2_phi<0] = np.pi + np.pi + pos2_phi[pos2_phi<0]
        mean_phi = np.average(pos2_phi, weights=q2r)
        var_phi2 = np.average((pos2_phi-mean_phi)**2, weights=q2r)
        r2  = Rpos(np.sqrt(var_phi2)).value
        if np.isnan(r1) or np.isnan(r2):
            continue

        sel1_phi = q1>thr_phi
        q1phi    = q1  [sel1_phi]
        pos1phi  = pos1[sel1_phi]
        sel2_phi = q2>thr_phi
        q2phi    = q2  [sel2_phi]
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
        q1z    = q1  [sel1_z]
        pos1z  = pos1[sel1_z]
        sel2_z = q2>thr_z
        q2z    = q2  [sel2_z]
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

        pos1_cart = []
        pos2_cart = []
        if r1 and phi1 and z1 and len(q1) and r2 and phi2 and z2 and len(q2):
            pos1_cart.append(r1 * np.cos(phi1))
            pos1_cart.append(r1 * np.sin(phi1))
            pos1_cart.append(z1)
            pos2_cart.append(r2 * np.cos(phi2))
            pos2_cart.append(r2 * np.sin(phi2))
            pos2_cart.append(z2)
        else: continue

        a_cart1 = np.array(pos1_cart)
        a_cart2 = np.array(pos2_cart)


        ## Tof convolution
        tof_sns = evt_tof.sensor_id.unique()
        evt_tof_exp_dist = []
        for s_id in tof_sns:
            tdc_conv    = tf.tdc_convolution(evt_tof, spe_resp, s_id, time_window)
            tdc_conv_df = tf.translate_charge_conv_to_wf_df(evt, s_id, tdc_conv)
            evt_tof_exp_dist.append(tdc_conv_df)
        evt_tof_exp_dist = pd.concat(evt_tof_exp_dist)

        ## Trying different thresholds in charge for the sensor that sees the first pe:
        for k, th in enumerate(timestamp_thr):
            evt_tof_exp_dist = evt_tof_exp_dist[evt_tof_exp_dist.charge > th/norm]
            min_id1, min_id2, min_t1, min_t2 = rf.find_first_times_of_coincidences(evt_sns, evt_tof_exp_dist, charge_range, DataSiPM_idx, evt_parts, evt_hits)

            min_t1 = min_t1*tof_bin_size/units.ps
            min_t2 = min_t2*tof_bin_size/units.ps

            min_pos1 = sensor_position(min_id1, sipms)
            min_pos2 = sensor_position(min_id2, sipms)

            ### Distance between interaction point and sensor detecting first photon
            dp1 = np.linalg.norm(a_cart1 - min_pos1)
            dp2 = np.linalg.norm(a_cart2 - min_pos2)

            delta_t = min_t2 - min_t1 + (dp1 - dp2)/ave_speed_in_LXe

            time_diff[k].append(delta_t)

        pos_cart1.append(a_cart1)
        pos_cart2.append(a_cart2)
        event_ids.append(evt)


a_time_diff = np.array(time_diff)
a_pos_cart1 = np.array(pos_cart1)
a_pos_cart2 = np.array(pos_cart2)
a_event_ids = np.array(event_ids)

np.savez(evt_file, time_diff=a_time_diff, pos_cart1=a_pos_cart1,
         pos_cart2=a_pos_cart2, event_ids=a_event_ids)

print(datetime.datetime.now())


print(f"Not a coincidence: {c0}")
print(f"Not passing threshold r = {c1}, phi = {c2}, z = {c3}, E = {c4}")
