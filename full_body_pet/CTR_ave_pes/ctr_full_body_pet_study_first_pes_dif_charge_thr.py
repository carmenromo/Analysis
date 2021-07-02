import sys
import argparse
import datetime
import tables         as tb
import numpy          as np
import pandas         as pd

import antea.database.load_db    as db
import antea.reco.reco_functions as rf
import antea.elec.tof_functions  as tf

from antea.utils.map_functions import load_map
from antea.io   .mc_io           import load_mchits
from antea.io   .mc_io           import load_mcparticles
from antea.io   .mc_io           import load_mcsns_response
from antea.io   .mc_io           import load_mcTOFsns_response
from antea.io   .mc_io           import read_sensor_bin_width_from_conf

from invisible_cities.core import system_of_units as units

"""
python ctr_full_body_phantom_paper_study_first_pes_dif_charge_thr.py 1255 1 4 4 4 2 /Users/carmenromoluque/nexus_petit_analysis/full-body-phantom-paper/h5_files/
full_body_phantom_paper /Users/carmenromoluque/Analysis/full_body_phantom_paper/r_table_full_body_phantom_paper_thr4pes.h5 /Users/carmenromoluque/Desktop/
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
    min_ts           = tof.groupby(['sensor_id'])[['time']].min().sort_values('time')
    mean_t_sel_sns   = min_ts[:num_sel_sns].time.mean()
    ids_sel_sns      = min_ts[:num_sel_sns].index.values
    evt_sns_sel_sns  = evt_sns[evt_sns.sensor_id.isin(-ids_sel_sns)]
    charges_sel_sns  = evt_sns_sel_sns.groupby(['sensor_id'])[['charge']].sum().values.T[0]
    sipms            = DataSiPM_idx.loc[evt_sns_sel_sns.sensor_id]
    sns_positions    = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).T
    weig_pos_sel_sns = np.average(                sns_positions, axis=0, weights=charges_sel_sns)
    weig_t_sel_sns   = np.average(min_ts[:num_sel_sns].time, axis=0, weights=charges_sel_sns)
    return ids_sel_sns, charges_sel_sns, mean_t_sel_sns, weig_t_sel_sns, weig_pos_sel_sns


def reconstruct_coincidences2(sns_response, tof_response, charge_range, DataSiPM_idx, particles, hits, num_of_init_pes):
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

    sns1, sns2, pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_ids, sns_positions, sns_charges, max_pos)

    tot_q1 = sum(q1)
    tot_q2 = sum(q2)

    sel1 = (tot_q1 > charge_range[0]) & (tot_q1 < charge_range[1])
    sel2 = (tot_q2 > charge_range[0]) & (tot_q2 < charge_range[1])
    if not sel1 or not sel2:
        return [], [], [], [], None, None, None, None, [], [], None, None, None, None, [], []

    ### TOF
    ids_tof_tot1 , ids_tof_tot2  = [], []
    mean_tof_tot1, mean_tof_tot2 = [], []
    weig_tof_tot1, weig_tof_tot2 = [], []
    weig_pos_tot1, weig_pos_tot2 = [], []
    for num in num_of_init_pes:
        ids_sel_sns1, _, mean_tof1, weig_tof1, weig_pos1 = find_selected_times_of_sensors(sns_response, tof_response, -sns1, num, DataSiPM_idx)
        ids_sel_sns2, _, mean_tof2, weig_tof2, weig_pos2 = find_selected_times_of_sensors(sns_response, tof_response, -sns2, num, DataSiPM_idx)
        ids_tof_tot1 .append(ids_sel_sns1)
        ids_tof_tot2 .append(ids_sel_sns2)
        mean_tof_tot1.append(mean_tof1)
        mean_tof_tot2.append(mean_tof2)
        weig_tof_tot1.append(weig_tof1)
        weig_tof_tot2.append(weig_tof2)
        weig_pos_tot1.append(weig_pos1)
        weig_pos_tot2.append(weig_pos2)

    true_pos1, true_pos2, true_t1, true_t2, _, _ = rf.find_first_interactions_in_active(particles, hits)

    if not len(true_pos1) or not len(true_pos2):
        print("Cannot find two true gamma interactions for this event")
        return [], [], [], [], None, None, None, None, [], [], None, None, None, None, [], []

    scalar_prod = true_pos1.dot(max_pos)
    if scalar_prod > 0:
        int_pos1 = pos1
        int_pos2 = pos2
        int_q1   = q1
        int_q2   = q2
        int_min1 = ids_tof_tot1
        int_min2 = ids_tof_tot2
        int_mean_tof1 = mean_tof_tot1
        int_mean_tof2 = mean_tof_tot2
        int_weig_tof1 = weig_tof_tot1
        int_weig_tof2 = weig_tof_tot2
        weig_pos1     = weig_pos_tot1
        weig_pos2     = weig_pos_tot2
    else:
        int_pos1 = pos2
        int_pos2 = pos1
        int_q1   = q2
        int_q2   = q1
        int_min1 = ids_tof_tot2
        int_min2 = ids_tof_tot1
        int_mean_tof1 = mean_tof_tot2
        int_mean_tof2 = mean_tof_tot1
        int_weig_tof1 = weig_tof_tot2
        int_weig_tof2 = weig_tof_tot1
        weig_pos1     = weig_pos_tot2
        weig_pos2     = weig_pos_tot1

    return int_pos1, int_pos2, int_q1, int_q2, true_pos1, true_pos2, true_t1, true_t2, int_min1, int_min2, int_mean_tof1, int_mean_tof2, int_weig_tof1, int_weig_tof2, weig_pos1, weig_pos2


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

evt_file  = f"{data_path}/full_body_ctr_number_pes_for_tof_{start}_{numb}_{thr_r}_{thr_phi}_{thr_z}_{thr_e}"
Rpos = load_map(rpos_file,
                group  = "Radius",
                node   = f"f{int(thr_r)}pes150bins",
                x_name = "PhiRms",
                y_name = "Rpos",
                u_name = "RposUncertainty")

### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

c0 = c1 = c2 = c3 = c4 = 0

### TOF elec parameters:
tau_sipm       = [100, 15000]
time_window    = 5000 #ps
time           = np.arange(0, time_window)
#time_bin       = 5 # ps
#time           = np.arange(0, 80000, time_bin)
#time           = time + (time_bin/2)
spe_resp, norm = tf.apply_spe_dist(time, tau_sipm)

timestamp_thr   = [0, 0.25, 0.5, 1.0]
num_of_init_pes = [1, 2, 3, 5, 8, 10, 12, 14, 16]

time_diff1 = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
time_diff2 = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
ave_t1     = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
ave_t2     = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
wei_t1     = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
wei_t2     = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
wpos1      = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
wpos2      = [[[] for k in range(len(timestamp_thr))] for j in range(len(num_of_init_pes))]
pos_cart1  = []
pos_cart2  = []
event_ids  = []

ave_speed_in_LXe = 0.210       # mm/ps
speed_in_vacuum  = 0.299792458 # mm/ps


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

    tof_bin_size = read_sensor_bin_width_from_conf(filename, tof=True)

    #sns_response_tof = load_mcTOFsns_response(filename)
    sns_response_tof = pd.read_hdf(filename, 'MC/tof_waveforms')
    particles        = load_mcparticles(filename)
    hits             = load_mchits(filename)

    events = particles.event_id.unique()

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()

    #charge_range = (0, 5000)
    charge_range = (1050, 1300)

    for evt in events[:]:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=2)
        if len(evt_sns) == 0:
            continue

        evt_parts = particles       [particles       .event_id == evt]
        evt_hits  = hits            [hits            .event_id == evt]
        evt_tof   = sns_response_tof[sns_response_tof.event_id == evt]

        pos1, pos2, q1, q2, true_pos1, true_pos2, true_t1, true_t2, sns1, sns2 = rf.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)

        if len(pos1) == 0 or len(pos2) == 0:
            c0 += 1
            continue

        q1   = np.array(q1)
        q2   = np.array(q2)
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

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

        ### Distance between interaction point and center of the geometry
        geo_center = np.array([0,0,0])
        dg1 = np.linalg.norm(a_cart1 - geo_center)
        dg2 = np.linalg.norm(a_cart2 - geo_center)

        times = evt_tof.time_bin.values * tof_bin_size / units.ps
        evt_tof['time'] = np.round(np.random.normal(times, 0)).astype(int)

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
            _, _, _, _, _, _, _, _, min_ids1, min_ids2, mean_tof1, mean_tof2, weig_tof1, weig_tof2, weig_pos1, weig_pos2 = reconstruct_coincidences2(evt_sns, evt_tof_exp_dist, charge_range, DataSiPM_idx, evt_parts, evt_hits, num_of_init_pes)
            for j, ipes in enumerate(num_of_init_pes):
                if mean_tof1[j] == None or weig_tof1[j] == None:
                    mean_t1 = mean_tof1[j]
                    weig_t1 = weig_tof1[j]
                else:
                    mean_t1 = mean_tof1[j]/units.ps
                    weig_t1 = weig_tof1[j]/units.ps

                if mean_tof2[j] == None or weig_tof2[j] == None:
                    mean_t2 = mean_tof2[j]
                    weig_t2 = weig_tof2[j]
                else:
                    mean_t2  = mean_tof2[j]/units.ps
                    weig_t2  = weig_tof2[j]/units.ps

                ### Distance between interaction point and sensor detecting first photon
                dp1 = np.linalg.norm(a_cart1 - weig_pos1[j])
                dp2 = np.linalg.norm(a_cart2 - weig_pos2[j])

                dg1 = np.linalg.norm(a_cart1)
                dg2 = np.linalg.norm(a_cart2)

                delta_t1 = mean_t2 - mean_t1 + (dp1 - dp2)/ave_speed_in_LXe + (dg1 - dg2)/speed_in_vacuum
                delta_t2 = weig_t2 - weig_t1 + (dp1 - dp2)/ave_speed_in_LXe + (dg1 - dg2)/speed_in_vacuum

                time_diff1[j][k].append(delta_t1)
                time_diff2[j][k].append(delta_t2)
                ave_t1    [j][k].append(mean_t1)
                ave_t2    [j][k].append(mean_t2)
                wei_t1    [j][k].append(weig_t1)
                wei_t2    [j][k].append(weig_t2)
                wpos1     [j][k].append(weig_pos1[j])
                wpos2     [j][k].append(weig_pos2[j])

        pos_cart1.append(a_cart1)
        pos_cart2.append(a_cart2)
        event_ids.append(evt)


a_time_diff1 = np.array(time_diff1)
a_time_diff2 = np.array(time_diff2)
a_ave_t1     = np.array(ave_t1)
a_ave_t2     = np.array(ave_t2)
a_wei_t1     = np.array(wei_t1)
a_wei_t2     = np.array(wei_t2)
a_wpos1      = np.array(wpos1)
a_wpos2      = np.array(wpos2)
a_pos_cart1  = np.array(pos_cart1)
a_pos_cart2  = np.array(pos_cart2)
a_event_ids  = np.array(event_ids)

np.savez(evt_file, time_diff1=a_time_diff1, time_diff2=a_time_diff2, ave_t1=a_ave_t1, ave_t2=a_ave_t2,
         wei_t1=a_wei_t1, wei_t2=a_wei_t2, wpos1=a_wpos1, wpos2=a_wpos2, pos_cart1=a_pos_cart1,
         pos_cart2=a_pos_cart2, event_ids=a_event_ids)

print(datetime.datetime.now())


print(f"Not a coincidence: {c0}")
print(f"Not passing threshold r = {c1}, phi = {c2}, z = {c3}, E = {c4}")
