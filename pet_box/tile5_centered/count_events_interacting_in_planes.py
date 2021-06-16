import sys
import math
import argparse
import datetime

import tables as tb
import numpy  as np
import pandas as pd

import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.elec.tof_functions    as tf
import antea.reco.mctrue_functions as mcf
import antea.io  .mc_io            as mcio

from antea.utils.map_functions import load_map
from invisible_cities.core     import system_of_units as units

""" To run this script
python count_events_interacting_in_planes.py 158 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/ PetBox_asymmetric_tile5centered_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
"""

print(datetime.datetime.now())

arguments     = pbf.parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
thr_ch_start  = arguments.thr_ch_start
thr_ch_nsteps = arguments.thr_ch_nsteps
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

def charges_in_coinc_plane(DataSiPM_idx, evt_sns):
    sipms         = DataSiPM_idx.loc[evt_sns.sensor_id]
    sns_ids       = sipms.index.astype('int64').values
    sns_pos       = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = evt_sns.charge.values
    sel           = sipms.Z.values>0
    coinc, detect = False, False
    if len(sns_ids[sel]):
        coinc = True
    if len(sns_ids[~sel]):
        detect = True
    return coinc, detect


evt_file   = f'{out_path}/count_evts_only_interacting_in_det_plane_{start}_{numb}'

event_ids1 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
event_ids2 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
event_ids3 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
event_ids4 = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
saved_evts = []
sns_evts   = []

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename = in_path + f'{file_name}.{number_str}.pet.h5'
    try:
        sns_response = mcio.load_mcsns_response(filename)
    except OSError:
        print(f'File {filename} does not exist')
        continue
    #print(f'file {number}')

    sns_positions = mcio.load_sns_positions(filename)
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    h5conf = mcio.load_configuration(filename)
    s_evts = int(h5conf[h5conf.param_key=='saved_events'].param_value.values[0])

    events = sns_response.event_id.unique()
    sns_evts.append(len(events))
    for evt in events:
        evt_sns   = sns_response[sns_response.event_id == evt]

        for j, th in enumerate(range(thr_ch_start, thr_ch_nsteps)):
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=th)
            if len(evt_sns) == 0:
                continue

            ch_coinc_plane, ch_det_plane = charges_in_coinc_plane(DataSiPM_idx, evt_sns)
            if ch_coinc_plane and ch_det_plane:
                event_ids1[th].append(evt)
            elif not ch_coinc_plane and ch_det_plane:
                event_ids2[th].append(evt)
            elif ch_coinc_plane and not ch_det_plane:
                event_ids3[th].append(evt)
            else:
                event_ids4[th].append(evt)

event_ids1_a = np.array([np.array(i) for i in event_ids1])
event_ids2_a = np.array([np.array(i) for i in event_ids2])
event_ids3_a = np.array([np.array(i) for i in event_ids3])
event_ids4_a = np.array([np.array(i) for i in event_ids4])
saved_evts_a = np.array(saved_evts)
sns_evts_a   = np.array(sns_evts)

np.savez(evt_file, event_ids1_0=event_ids1_a[0], event_ids1_1=event_ids1_a[1], event_ids1_2=event_ids1_a[2],
         event_ids1_3=event_ids1_a[3], event_ids1_4=event_ids1_a[4], event_ids1_5=event_ids1_a[5],
         event_ids2_0=event_ids2_a[0], event_ids2_1=event_ids2_a[1], event_ids2_2=event_ids2_a[2],
         event_ids2_3=event_ids2_a[3], event_ids2_4=event_ids2_a[4], event_ids2_5=event_ids2_a[5],
         event_ids3_0=event_ids3_a[0], event_ids3_1=event_ids3_a[1], event_ids3_2=event_ids3_a[2],
         event_ids3_3=event_ids3_a[3], event_ids3_4=event_ids3_a[4], event_ids3_5=event_ids3_a[5],
         event_ids4_0=event_ids4_a[0], event_ids4_1=event_ids4_a[1], event_ids4_2=event_ids4_a[2],
         event_ids4_3=event_ids4_a[3], event_ids4_4=event_ids4_a[4], event_ids4_5=event_ids4_a[5],
         saved_evts=saved_evts_a, sns_evts=sns_evts_a)

print(datetime.datetime.now())
