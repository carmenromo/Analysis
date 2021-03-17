import sys
import math
import argparse
import tables as tb
import numpy  as np
import pandas as pd
import datetime

import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

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

evts_parts = []
evts_hits  = []
evts_sns   = []

evt_file   = f'{out_path}/check_tables_length_{start}_{numb}'

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

    events1 = mcparticles .event_id.unique()
    events2 = mchits      .event_id.unique()
    events3 = sns_response.event_id.unique()

    evts_parts.append(events1)
    evts_hits .append(events2)
    evts_sns  .append(events3)

evts_parts = np.array(evts_parts)
evts_hits  = np.array(evts_hits)
evts_sns   = np.array(evts_sns)

np.savez(evt_file, events1=events1, events2=events2, events3=events3)
print(datetime.datetime.now())
