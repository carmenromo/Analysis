import os
import sys
import math
import datetime
import tables         as tb
import numpy          as np
import pandas         as pd
import daq_functions  as dq

from antea.utils.table_functions import load_rpos
import antea.database.load_db    as db

print(datetime.datetime.now())


data_path  = f"/data5/users/carmenromo/DAQ_ANTEA/"
evt_file   = f"/data5/users/carmenromo/DAQ_ANTEA/data_reco_pos/full_ring_ctr_daq_test2"

charge_range = (1000, 1500)
thr_r   = 4
thr_phi = 4
thr_z   = 4
thr_e   = 2

rpos_file = f"/data5/users/carmenromo/DAQ_ANTEA/r_sigma_phi_table_irad165mm_depth3cm_thr{thr_r}pes_no_compton2.h5"
Rpos = load_rpos(rpos_file, group = "Radius", node = f"f{thr_r}pes200bins")

### read sensor positions from database
DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

time_diff1 = []
time_diff2 = []
pos_cart1  = []
pos_cart2  = []
event_ids  = []

c0 = c1 = c2 = c3 = c4 = 0
bad = 0

ave_speed_in_LXe = 0.210 # mm/ps
speed_in_vacuum  = 0.299792458 # mm/ps

filename = f"{data_path}/DAQ_OF_7mm_CUBE_RAW.h5"
data     = pd.read_hdf(filename, key='raw')
data     = data.sort_values('in_time')

data.loc[(data.in_time.shift() < data.in_time - 1000), 'group'] = 1
data['group'] = data['group'].cumsum().ffill().fillna(0)

events = data.group.unique()

for evt in events[:]:
    evt_sns = data[data.group == evt]
    sns1, sns2, pos1, pos2, q1, q2, min_id1, min_id2, min_t1, min_t2 = dq.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx)

    if len(pos1) == 0 or len(pos2) == 0:
        continue

    q1   = np.array(q1)
    q2   = np.array(q2)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    sns1 = np.array(sns1)
    sns2 = np.array(sns2)

    ## Calculate R
    r1 = r2 = None

    q1r, pos1r, _ = dq.threshold_filter(thr_r, q1, pos1, sns1)
    q2r, pos2r, _ = dq.threshold_filter(thr_r, q2, pos2, sns2)
    if len(pos1r) == 0 or len(pos2r) == 0:
        c1 += 1
        continue

    var_phi1 = dq.get_var_phi(pos1r, q1r)
    var_phi2 = dq.get_var_phi(pos2r, q2r)
    r1 = Rpos(np.sqrt(var_phi1)).value
    r2 = Rpos(np.sqrt(var_phi2)).value

    q1phi, pos1phi, _ = dq.threshold_filter(thr_phi, q1, pos1, sns1)
    q2phi, pos2phi, _ = dq.threshold_filter(thr_phi, q2, pos2, sns2)
    if len(q1phi) == 0 or len(q2phi) == 0:
        c2 += 1
        continue

    phi1 = phi2 = None
    reco_cart_pos = np.average(pos1phi, weights=q1phi, axis=0)
    phi1 = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])
    reco_cart_pos = np.average(pos2phi, weights=q2phi, axis=0)
    phi2 = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])


    q1z, pos1z, _ = dq.threshold_filter(thr_z, q1, pos1, sns1)
    q2z, pos2z, _ = dq.threshold_filter(thr_z, q2, pos2, sns2)
    if len(q1z) == 0 or len(q2z) == 0:
        c3 += 1
        continue

    z1 = z2 = None
    reco_cart_pos = np.average(pos1z, weights=q1z, axis=0)
    z1 = reco_cart_pos[2]
    reco_cart_pos = np.average(pos2z, weights=q2z, axis=0)
    z2 = reco_cart_pos[2]

    q1e, _, ids1 = dq.threshold_filter(thr_e, q1, pos1, sns1)
    q2e, _, ids2 = dq.threshold_filter(thr_e, q2, pos2, sns2)
    if len(q1e) == 0 or len(q2e) == 0:
        c4 += 1
        continue

    pos1_cart = []
    pos2_cart = []
    if r1 and not np.isnan(r1) and phi1 and z1 and len(q1) and r2 and not np.isnan(r2) and phi2 and z2 and len(q2) and min_t1 and min_t2:
        pos1_cart.append(r1 * np.cos(phi1))
        pos1_cart.append(r1 * np.sin(phi1))
        pos1_cart.append(z1)
        pos2_cart.append(r2 * np.cos(phi2))
        pos2_cart.append(r2 * np.sin(phi2))
        pos2_cart.append(z2)
    else: continue
    
    ### CAREFUL, I AM BLENDING THE EVENTS!!!
    ##if evt%2 == 0:
    ##    a_cart1 = np.array(pos1_cart)
    ##   a_cart2 = np.array(pos2_cart)
    ##    min_t1  = min_t1/0.001
    ##    min_t2  = min_t2/0.001
    ##else:
    ##    a_cart1 = np.array(pos2_cart)
    ##    a_cart2 = np.array(pos1_cart)
    ##    min_t1  = min_t2/0.001 
    ##    min_id1 = min_id2
    ##    min_t2  = min_t1/0.001
    ##    min_id2 = min_id1

    a_cart1 = np.array(pos1_cart)
    a_cart2 = np.array(pos2_cart)
    min_t1  = min_t1/0.001
    min_t2  = min_t2/0.001

    ### Distance between interaction point and sensor detecting first photon
    m_id1    = DataSiPM[DataSiPM.SensorID==min_id1]
    sns_pos1 = np.array([m_id1.X.values[0], m_id1.Y.values[0], m_id1.Z.values[0]])
    m_id2    = DataSiPM[DataSiPM.SensorID==min_id2]
    sns_pos2 = np.array([m_id2.X.values[0], m_id2.Y.values[0], m_id2.Z.values[0]])

    dp1 = np.linalg.norm(a_cart1 - sns_pos1)
    dp2 = np.linalg.norm(a_cart2 - sns_pos2)

    ### Distance between interaction point and center of the geometry
    geo_center = np.array([0,0,0])
    dg1 = np.linalg.norm(a_cart1 - geo_center)
    dg2 = np.linalg.norm(a_cart2 - geo_center)

    delta_t1 = 1/2 *(min_t2 - min_t1 + (dp1 - dp2)/ave_speed_in_LXe)
    delta_t2 = 1/2 *(min_t2 - min_t1)

    print(delta_t1)
    print(delta_t2)
    time_diff1.append(delta_t1)
    time_diff2.append(delta_t2)
    pos_cart1 .append(a_cart1)
    pos_cart2 .append(a_cart2)
    event_ids .append(evt)

a_time_diff1 = np.array(time_diff1)
a_time_diff2 = np.array(time_diff2)
a_pos_cart1  = np.array(pos_cart1)
a_pos_cart2  = np.array(pos_cart2)
a_event_ids  = np.array(event_ids)

np.savez(evt_file, a_time_diff1=a_time_diff1, a_time_diff2=a_time_diff2, a_pos_cart1=a_pos_cart1, a_pos_cart2=a_pos_cart2, a_event_ids=a_event_ids)

print(datetime.datetime.now())
