import sys
import math
import argparse
import tables as tb
import numpy  as np
import pandas as pd
import datetime

#import pet_box_functions as pbf

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

import antea.io.mc_io as mcio

""" To run this script
python 1_pet_box_build_z_map_both_planes.py 0 1 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5/
PetBox_asymmetric_tile5centered_HamamatsuVUV /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
"""

print(datetime.datetime.now())

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

true_pos1 = []
true_pos2 = []
events1   = []
events2   = []
charges1  = []
charges2  = []
ids1      = []
ids2      = []

#int_area = np.array([22, 23, 24, 25, 26, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 73, 74, 75, 76, 77,
#                     33, 34, 35, 36, 43, 46, 53, 56, 63, 64, 65, 66, 44, 45, 54, 55])

evt_file   = f'{out_path}/pet_box_true_info_teflon_block_fluct_{start}_{numb}'

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = f'{in_path}/{file_name}.{number_str}.pet.h5'
    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        print(f'File {true_file} does not exist')
        continue
    print(f'Analyzing file {true_file}')

    mcparticles   = mcio.load_mcparticles   (true_file)
    mchits        = mcio.load_mchits        (true_file)
    sns_response  = mcio.load_mcsns_response(true_file)

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_sns   = sns_response[sns_response.event_id == evt]
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]

        th = 2
        fluct_evt_sns = snsf.apply_charge_fluctuation(evt_sns, DataSiPM_pb_idx)
        fluct_evt_sns = rf.find_SiPMs_over_threshold(fluct_evt_sns, threshold=th)

        phot, true_pos_phot = mcf.select_photoelectric(evt_parts, evt_hits)

        if phot:
            for pos in true_pos_phot:
                if pos[2]<0 and len(fluct_evt_sns.charge.values[fluct_evt_sns.sensor_id.values<100]):
                    true_pos1.append(pos)
                    events1  .append(evt)
                    charges1 .append(fluct_evt_sns.charge   .values[fluct_evt_sns.sensor_id.values<100])
                    ids1     .append(fluct_evt_sns.sensor_id.values[fluct_evt_sns.sensor_id.values<100])
                elif pos[2]>0 and len(fluct_evt_sns.charge.values[fluct_evt_sns.sensor_id.values>100]):
                    true_pos2.append(pos)
                    events2  .append(evt)
                    charges2 .append(fluct_evt_sns.charge   .values[fluct_evt_sns.sensor_id.values>100])
                    ids2     .append(fluct_evt_sns.sensor_id.values[fluct_evt_sns.sensor_id.values>100])

true_pos1_a = np.array(true_pos1)
true_pos2_a = np.array(true_pos2)
events1_a   = np.array(events1)
events2_a   = np.array(events2)
charges1_a  = np.array(charges1)
charges2_a  = np.array(charges2)
ids1_a      = np.array(ids1)
ids2_a      = np.array(ids2)

np.savez(evt_file, true_pos1=true_pos1_a, true_pos2=true_pos2_a, events1=events1_a, events2=events2_a, charges1=charges1_a, charges2=charges2_a, ids1=ids1_a, ids2=ids2_a)

print(datetime.datetime.now())
