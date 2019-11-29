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
evt_file   = f"/data5/users/carmenromo/DAQ_ANTEA/data_reco_pos/full_ring_reco_pos_charge_daq_allevt"

charge_range = (1000, 1400)
thr_r   = 4
thr_phi = 4
thr_z   = 4
thr_e   = 2

#rpos_file = f"/data5/users/carmenromo/DAQ_ANTEA/r_sigma_phi_table_irad165mm_depth3cm_thr{thr_r}pes_no_compton2.h5"
rpos_file = '/data5/users/carmenromo/DAQ_ANTEA/r_table_iradius165mm_depth3cm_pitch7mm_new_h5_clean_thr4pes.h5'
Rpos = load_rpos(rpos_file, group = "Radius", node = f"f{thr_r}pes200bins")

### read sensor positions from database
DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

reco_r1, reco_phi1, reco_z1 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

charges1, charges2 = [], []
sensors1, sensors2 = [], []

sns_response1, sns_response2    = [], []
first_sipm1, first_sipm2        = [], []
first_time1, first_time2        = [], []
touched_sipms1, touched_sipms2  = [], []

event_ids = []

c0 = c1 = c2 = c3 = c4 = 0
bad = 0

filename = f"{data_path}/DAQ_OF_7mm_CUBE_RAW.h5"
data     = pd.read_hdf(filename, key='raw')
data     = data.sort_values('in_time')

data.loc[(data.in_time.shift() < data.in_time - 1000), 'group'] = 1
data['group'] = data['group'].cumsum().ffill().fillna(0)

events = data.group.unique()
print(len(events))
for evt in events[:]:
    #print(data[data.group == evt])
    evt_sns = data[data.group == evt]
    sns1, sns2, pos1, pos2, q1, q2, min1, min2, t1, t2 = dq.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx)

    if len(pos1) == 0 or len(pos2) == 0:
        c0 += 1
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

    event_ids     .append(evt)
    reco_r1       .append(r1)
    reco_phi1     .append(phi1)
    reco_z1       .append(z1)
    charges1      .append(q1e)
    sensors1      .append(ids1)
    sns_response1 .append(sum(q1e))
    touched_sipms1.append(len(q1e))
    first_sipm1   .append(min1)
    first_time1   .append(t1)

    reco_r2       .append(r2)
    reco_phi2     .append(phi2)
    reco_z2       .append(z2)
    charges2      .append(q2e)
    sensors2      .append(ids2)
    sns_response2 .append(sum(q2e))
    touched_sipms2.append(len(q2e))
    first_sipm2   .append(min2)
    first_time2   .append(t2)

a_reco_r1        = np.array(reco_r1)
a_reco_phi1      = np.array(reco_phi1)
a_reco_z1        = np.array(reco_z1)
a_charges1       = np.array(charges1)
a_sensors1       = np.array(sensors1)
a_sns_response1  = np.array(sns_response1)
a_touched_sipms1 = np.array(touched_sipms1)
a_first_sipm1    = np.array(first_sipm1)
a_first_time1    = np.array(first_time1)

a_reco_r2        = np.array(reco_r2)
a_reco_phi2      = np.array(reco_phi2)
a_reco_z2        = np.array(reco_z2)
a_charges2       = np.array(charges2)
a_sensors2       = np.array(sensors2)
a_sns_response2  = np.array(sns_response2)
a_touched_sipms2 = np.array(touched_sipms2)
a_first_sipm2    = np.array(first_sipm2)
a_first_time2    = np.array(first_time2)
a_event_ids      = np.array(event_ids)

np.savez(evt_file, a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1, a_reco_r2=a_reco_r2, 
         a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2, a_charges1=a_charges1, a_charges2=a_charges2, 
         a_sensors1=a_sensors1, a_sensors2=a_sensors2, a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2, 
         a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2, a_first_sipm1=a_first_sipm1, 
         a_first_time1=a_first_time1, a_first_sipm2=a_first_sipm2, a_first_time2=a_first_time2, a_event_ids=a_event_ids)

print(datetime.datetime.now())
print('')
print('Not a coincidence: {}'.format(c0))
print('Not passing threshold r = {}, phi = {}, z = {}, E = {}'.format(c1, c2, c3, c4))
