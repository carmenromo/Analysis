import sys
import argparse
import numpy  as np
import pandas as pd

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import reco_functions as rf2

from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcparticles


### read sensor positions from database
#DataSiPM     = db.DataSiPM('petalo', 0) # ring
DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
DataSiPM_idx = DataSiPM.set_index('SensorID')

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"     )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize" )
    parser.add_argument('thr_r'      , type = int, help = "threshold in r coordinate"  )
    parser.add_argument('events_path',             help = "input files path"           )
    parser.add_argument('file_name'  ,             help = "name of input files"        )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()

arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
threshold  = arguments.thr_r
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path

evt_file  = f"{data_path}/full_body_r_map_compton_efrom0.4_{start}_{numb}_{threshold}"

true_r1, true_r2   = [], []
var_phi1, var_phi2 = [], []
var_z1, var_z2     = [], []

touched_sipms1, touched_sipms2 = [], []

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

    sel_df = rf.find_SiPMs_over_threshold(sns_response, threshold)

    particles = load_mcparticles(filename)
    hits      = load_mchits     (filename)
    events    = particles.event_id.unique()

    for evt in events:
        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = rf2.true_photoelect(evt_parts, evt_hits, compton=True)
        if not select: continue

        waveforms = sel_df[sel_df.event_id == evt]
        if len(waveforms) == 0: continue

        _, _, pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(waveforms, true_pos, DataSiPM_idx)

        if len(pos1) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            _, var_phi = rf.phi_mean_var(pos_phi, q1)

            pos_z  = np.array(pos1)[:,2]
            mean_z = np.average(pos_z, weights=q1)
            var_z  = np.average((pos_z-mean_z)**2, weights=q1)

            reco_cart = np.average(pos1, weights=q1, axis=0)
            r = np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2)

            true_r1       .append(r)
            var_phi1      .append(var_phi)
            var_z1        .append(var_z)
            touched_sipms1.append(len(pos1))
        else:
            var_phi1      .append(1.e9)
            var_z1        .append(1.e9)
            touched_sipms1.append(1.e9)
            true_r1       .append(1.e9)

        if len(pos2) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            _, var_phi = rf.phi_mean_var(pos_phi, q2)

            pos_z  = np.array(pos2)[:,2]
            mean_z = np.average(pos_z, weights=q2)
            var_z  = np.average((pos_z-mean_z)**2, weights=q2)

            reco_cart = np.average(pos2, weights=q2, axis=0)
            r = np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2)

            true_r2       .append(r)
            var_phi2      .append(var_phi)
            var_z2        .append(var_z)
            touched_sipms2.append(len(pos2))
        else:
            var_phi2      .append(1.e9)
            var_z2        .append(1.e9)
            touched_sipms2.append(1.e9)
            true_r2       .append(1.e9)

a_true_r1  = np.array(true_r1)
a_true_r2  = np.array(true_r2)
a_var_phi1 = np.array(var_phi1)
a_var_phi2 = np.array(var_phi2)
a_var_z1   = np.array(var_z1)
a_var_z2   = np.array(var_z2)

a_touched_sipms1 = np.array(touched_sipms1)
a_touched_sipms2 = np.array(touched_sipms2)

np.savez(evt_file, a_true_r1=a_true_r1, a_true_r2=a_true_r2, a_var_phi1=a_var_phi1, a_var_phi2=a_var_phi2, a_var_z1=a_var_z1, a_var_z2=a_var_z2, a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2)
