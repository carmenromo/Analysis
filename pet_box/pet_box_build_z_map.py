import sys
import math
import argparse
import tables as tb
import numpy  as np
import pandas as pd
import datetime

import pet_box_functions as pbf

import antea.reco.reco_functions as rf

from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions


print(datetime.datetime.now())

arguments     = pbf.parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
thr_ch_start  = arguments.thr_ch_start
thr_ch_nsteps = arguments.thr_ch_nsteps
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

true_z = [[] for i in range(thr_ch_start, thr_ch_nsteps)]
var_r  = [[] for i in range(thr_ch_start, thr_ch_nsteps)]

evt_file   = f'{out_path}/pet_box_build_z_map_{start}_{numb}_{thr_ch_start}_{thr_ch_nsteps}'

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = f'{in_path}/{file_name}.{number_str}.pet.h5'
    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        print(f'File {true_file} does not exist')
        continue
    print(f'Analyzing file {true_file}')

    mcparticles   = load_mcparticles   (true_file)
    mchits        = load_mchits        (true_file)
    sns_response  = load_mcsns_response(true_file)
    sns_positions = load_sns_positions (true_file)
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]

        phot, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)

        if phot and true_pos[0][2]<0:
            for n_th, threshold in enumerate(range(thr_ch_start, thr_ch_nsteps)):
                evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=n_th)

                ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
                if sum(qs) == 0:
                    continue

                pos_r  = np.array([np.sqrt(p[0]**2 + p[1]**2) for p in pos])
                mean_r = np.average(pos_r, weights=qs)
                var_rs = np.average((pos_r - mean_r)**2, weights=qs)

                var_r [n_th].append(var_rs)
                true_z[n_th].append(true_pos[0][2])

for i in range(thr_ch_nsteps):
    true_z[i] = np.array(true_z[i])
    var_r [i] = np.array(var_r [i])

np.savez(evt_file, a_true_z=true_z, a_var_r=var_r)

print(datetime.datetime.now())
