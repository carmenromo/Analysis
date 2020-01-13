import sys
import datetime
import sc_utils
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

"""
Example of calling this script:

python ctr_ring_no_true_info.py 0 1 6 0 /data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton/analysis/data_ring full_ring_iradius165mm_z140mm_depth3cm_pitch7mm /data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton/analysis/ 4_data_crt_no_compton irad165mm_depth3cm
"""

arguments  = sc_utils.parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
nsteps     = arguments.n_steps
thr_start  = arguments.thr_start
eventsPath = arguments.events_path
file_name  = arguments.file_name
base_path  = arguments.base_path
data_path  = arguments.data_path
identifier = arguments.identifier

data_path  = f"{base_path}/{data_path}"
evt_file   = f"{data_path}/full_ring_{identifier}_crt_{start}_{numb}"

thr_r   = 4
thr_phi = 5
thr_z   = 4
thr_e   = 2

def sensor_position(s_id, sipms):
    xpos = sipms[sipms.ChannelID==s_id].X.unique()[0]
    ypos = sipms[sipms.ChannelID==s_id].Y.unique()[0]
    zpos = sipms[sipms.ChannelID==s_id].Z.unique()[0]
    return np.array([xpos, ypos, zpos])

rpos_file = f"{base_path}/r_sigma_phi_table_{identifier}_thr{thr_r}pes_no_compton.h5"
Rpos      = ats.load_rpos(rpos_file, group="Radius", node=f"f4pes150bins")

DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

c0 = c1 = c2 = c3 = c4 = 0

time_diff = []
pos_cart1 = []
pos_cart2 = []
event_ids = []

ave_speed_in_LXe = 0.210 # mm/ps
speed_in_vacuum  = 0.299792458 # mm/ps

### TOF elec parameters:
SIPM        = {'n_sipms':3500, 'first_sipm':1000, 'tau_sipm':[100,15000]}
n_sipms     = SIPM['n_sipms']
first_sipm  = SIPM['first_sipm']
tau_sipm    = SIPM['tau_sipm']
TE_range    = [0.25]
TE_TDC      = TE_range[0]
time_window = 10000
time_bin    = 5
time        = np.arange(0, 80000, time_bin)
spe_resp    = tf.spe_dist(tau_sipm, time)

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

        pos1, pos2, q1, q2, _, _, _, _ = rf.select_coincidences(evt_sns, evt_tof, charge_range, DataSiPM_idx, evt_parts, evt_hits)

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

        tdc_conv_table = tf.tdc_convolution(evt_tof, spe_resp, time_window, n_sipms, first_sipm, TE_TDC)
        evt_tof_exp_dist = tf.translate_charge_matrix_to_wf_df(evt, tdc_conv_table, first_sipm)
        print(evt_sns, evt_tof_exp_dist)
        min_id1, min_id2, min_tof1, min_tof2 = rf.find_first_times_of_coincidences(evt_sns, evt_tof_exp_dist, charge_range, DataSiPM_idx, evt_parts, evt_hits)

        ### CAREFUL, I AM BLENDING THE EVENTS!!!                                                                                                                              
        if evt%2 == 0:
            a_cart1   = np.array(pos1_cart)
            a_cart2   = np.array(pos2_cart)
            min_t1    = min_tof1*tof_bin_size/units.ps
            min_t2    = min_tof2*tof_bin_size/units.ps
            min_pos1 = sensor_position(min_id1, sipms)
            min_pos2 = sensor_position(min_id2, sipms)
        else:
            a_cart1   = np.array(pos2_cart)
            a_cart2   = np.array(pos1_cart)
            min_t1    = min_tof2*tof_bin_size/units.ps
            min_t2    = min_tof1*tof_bin_size/units.ps
            min_pos1 = sensor_position(min_id2, sipms)
            min_pos2 = sensor_position(min_id1, sipms)


        ### Distance between interaction point and sensor detecting first photon
        dp1 = np.linalg.norm(a_cart1 - min_pos1)
        dp2 = np.linalg.norm(a_cart2 - min_pos2)

        ### Distance between interaction point and center of the geometry
        geo_center = np.array([0,0,0])
        dg1 = np.linalg.norm(a_cart1 - geo_center)
        dg2 = np.linalg.norm(a_cart2 - geo_center)

        delta_t = 1/2 *(min_t2 - min_t1 + (dp1 - dp2)/ave_speed_in_LXe)

        time_diff.append(delta_t)
        pos_cart1.append(a_cart1)
        pos_cart2.append(a_cart2)
        event_ids.append(evt)
        
        print(delta_t)

a_time_diff = np.array(time_diff)
a_pos_cart1 = np.array(pos_cart1)
a_pos_cart2 = np.array(pos_cart2)
a_event_ids = np.array(event_ids)

np.savez(evt_file, time_diff=a_time_diff, pos_cart1=a_pos_cart1, pos_cart2=a_pos_cart2, event_ids=a_event_ids)

print(datetime.datetime.now())


