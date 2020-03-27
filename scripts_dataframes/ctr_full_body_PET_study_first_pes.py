import sys
import datetime
import argparse
import tables         as tb
import numpy          as np
import pandas         as pd
import analysis_utils as ats

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.elec.tof_functions    as tf

from antea.utils.table_functions import load_rpos
from antea.io   .mc_io           import load_mchits
from antea.io   .mc_io           import load_mcparticles
from antea.io   .mc_io           import load_mcsns_response
from antea.io   .mc_io           import load_mcTOFsns_response
from antea.io   .mc_io           import read_sensor_bin_width_from_conf

from invisible_cities.core import system_of_units as units

print(datetime.datetime.now())


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
    parser.add_argument('base_path'  ,             help = "Path of the conf"           )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()

arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
thr_r      = arguments.thr_r
thr_phi    = arguments.thr_phi
thr_z      = arguments.thr_z
thr_e      = arguments.thr_e
eventsPath = arguments.events_path
file_name  = arguments.file_name
base_path  = arguments.base_path
data_path  = arguments.data_path

evt_file  = f"{base_path}/{data_path}/full_body_ctr_number_pes_for_tof_{start}_{numb}_{thr_r}_{thr_phi}_{thr_z}_{thr_e}"

rpos_file = f"{base_path}/r_table_iradius165mm_depth3cm_pitch7mm_new_h5_clean_thr{thr_r}pes.h5"
Rpos = load_rpos(rpos_file, group = "Radius", node  = f"f{thr_r}pes200bins")


DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

c0 = c1 = c2 = c3 = c4 = 0

num_of_init_pes = [1, 2, 3, 5, 8, 10]

time_diff1 = [[] for j in range(len(num_of_init_pes))]
time_diff2 = [[] for j in range(len(num_of_init_pes))]
ave_t1     = [[] for j in range(len(num_of_init_pes))]
ave_t2     = [[] for j in range(len(num_of_init_pes))]
wei_t1     = [[] for j in range(len(num_of_init_pes))]
wei_t2     = [[] for j in range(len(num_of_init_pes))]
wpos1      = [[] for j in range(len(num_of_init_pes))]
wpos2      = [[] for j in range(len(num_of_init_pes))]
pos_cart1  = []
pos_cart2  = []
event_ids  = []

ave_speed_in_LXe = 0.210       # mm/ps
speed_in_vacuum  = 0.299792458 # mm/ps


for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
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

    h5f = tb.open_file(filename, mode='r')
    tof_bin_size = read_sensor_bin_width_from_conf(h5f)
    h5f.close()

    sns_response_tof = load_mcTOFsns_response(filename)
    particles        = load_mcparticles(filename)
    hits             = load_mchits(filename)

    events = particles.event_id.unique()

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()

    charge_range = (1000, 1400)

    for evt in events[:]:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=2)
        if len(evt_sns) == 0:
            continue

        evt_parts = particles[particles.event_id       == evt]
        evt_hits  = hits[hits.event_id                 == evt]
        evt_tof   = sns_response_tof[sns_response_tof.event_id == evt]
        
        pos1, pos2, q1, q2, _, _, _, _, min_ids1, min_ids2, mean_tof1, mean_tof2, weig_tof1, weig_tof2, weig_pos1, weig_pos2 = rf.reconstruct_coincidences2(evt_sns, evt_tof, charge_range, DataSiPM_idx, evt_parts, evt_hits, num_of_init_pes)

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
        diff_sign = min(pos1_phi ) < 0 < max(pos1_phi)
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

        a_cart1   = np.array(pos1_cart)
        a_cart2   = np.array(pos2_cart)

        ### Distance between interaction point and center of the geometry
        geo_center = np.array([0,0,0])
        dg1 = np.linalg.norm(a_cart1 - geo_center)
        dg2 = np.linalg.norm(a_cart2 - geo_center)

        for j,k in enumerate(num_of_init_pes):
            mean_t1  = mean_tof1[j]*tof_bin_size/units.ps
            mean_t2  = mean_tof2[j]*tof_bin_size/units.ps
            weig_t1  = weig_tof1[j]*tof_bin_size/units.ps
            weig_t2  = weig_tof2[j]*tof_bin_size/units.ps

            ### Distance between interaction point and sensor detecting first photon
            dp1 = np.linalg.norm(a_cart1 - weig_pos1[j])
            dp2 = np.linalg.norm(a_cart2 - weig_pos2[j])


            delta_t1 = 1/2 *(mean_t2 - mean_t1 + (dp1 - dp2)/ave_speed_in_LXe)
            delta_t2 = 1/2 *(weig_t2 - weig_t1 + (dp1 - dp2)/ave_speed_in_LXe)

            #print(delta_t1)
            #print(delta_t2)

            time_diff1[j].append(delta_t1)
            time_diff2[j].append(delta_t2)
            ave_t1    [j].append(mean_t1)
            ave_t2    [j].append(mean_t2)
            wei_t1    [j].append(weig_t1)
            wei_t2    [j].append(weig_t2)
            wpos1     [j].append(weig_pos1[j])
            wpos2     [j].append(weig_pos2[j])

        pos_cart1.append(a_cart1)
        pos_cart2.append(a_cart2)
        event_ids.append(evt)


a_time_diff1 = np.array(time_diff1)
a_time_diff2 = np.array(time_diff2)
a_ave_t1 = np.array(ave_t1)
a_ave_t2 = np.array(ave_t2)
a_wei_t1 = np.array(wei_t1)
a_wei_t2 = np.array(wei_t2)
a_wpos1 = np.array(wpos1)
a_wpos2 = np.array(wpos2)
a_pos_cart1  = np.array(pos_cart1)
a_pos_cart2  = np.array(pos_cart2)
a_event_ids  = np.array(event_ids)

np.savez(evt_file, time_diff1=a_time_diff1, time_diff2=a_time_diff2, ave_t1=a_ave_t1, ave_t2=a_ave_t2,
         wei_t1=a_wei_t1, wei_t2=a_wei_t2, wpos1=a_wpos1, wpos2=a_wpos2, pos_cart1=a_pos_cart1,
         pos_cart2=a_pos_cart2, event_ids=a_event_ids)

print(datetime.datetime.now())


