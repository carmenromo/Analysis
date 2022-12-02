import sys
import argparse
import numpy  as np
import pandas as pd

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import reco_functions              as rf2
import antea.mcsim.sensor_functions as snsf

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

#def load_sens_pos(file_name):
#    sens_pos = pd.read_hdf(file_name, 'MC/sensor_positions')
#    return sens_pos

arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
threshold  = arguments.thr_r
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path

evt_file  = f"{data_path}/full_body_r_map_fluct_ch_{start}_{numb}_{threshold}"

true_r1, true_r2   = [], []
var_phi1, var_phi2 = [], []
var_z1, var_z2     = [], []
mean_phi1, mean_phi2 = [], []
mean_z1, mean_z2     = [], []

evt_ids, num_phot_evts = [], []
charges1, charges2 = [], []

touched_sipms1, touched_sipms2 = [], []

for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    #filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    filename  = f"{eventsPath}/{file_name}.{number}.h5"
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
    fluct_sns_response = snsf.apply_charge_fluctuation(sel_df, DataSiPM_idx)

    particles = load_mcparticles(filename)
    hits      = load_mchits     (filename)
    events    = particles.event_id.unique()

    #DataSiPM     = load_sens_pos(filename)
    #DataSiPM     = DataSiPM.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    #DataSiPM_idx = DataSiPM.set_index('SensorID')

    for evt in events:
        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits) #rf2.true_photoelect(evt_parts, evt_hits)
        if not select: continue

        sns_resp = fluct_sns_response[fluct_sns_response.event_id == evt]
        if len(sns_resp) == 0: continue

        _, _, pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(sns_resp, true_pos, DataSiPM_idx)

        evt_ids      .append(evt)
        num_phot_evts.append(len(true_pos))

        if len(pos1) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            mean_phi, var_phi = rf.phi_mean_var(pos_phi, q1)

            pos_z  = np.array(pos1)[:,2]
            mean_z = np.average(pos_z, weights=q1)
            var_z  = np.average((pos_z-mean_z)**2, weights=q1)

            r = np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2)

            true_r1       .append(r)
            var_phi1      .append(var_phi)
            var_z1        .append(var_z)
            touched_sipms1.append(len(pos1))
            mean_phi1     .append(mean_phi)
            mean_z1       .append(mean_z)
            charges1      .append(sum(q1))
        
        else:
            var_phi1      .append(1.e9)
            var_z1        .append(1.e9)
            touched_sipms1.append(1.e9)
            true_r1       .append(1.e9)
            mean_phi1     .append(1.e9)
            mean_z1       .append(1.e9)
            charges1      .append(1.e9)

        if len(pos2) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            mean_phi, var_phi = rf.phi_mean_var(pos_phi, q2)

            pos_z  = np.array(pos2)[:,2]
            mean_z = np.average(pos_z, weights=q2)
            var_z  = np.average((pos_z-mean_z)**2, weights=q2)

            r = np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2)

            true_r2       .append(r)
            var_phi2      .append(var_phi)
            var_z2        .append(var_z)
            touched_sipms2.append(len(pos2))
            mean_phi2     .append(mean_phi)
            mean_z2       .append(mean_z)
            charges2      .append(sum(q2))
        else:
            var_phi2      .append(1.e9)
            var_z2        .append(1.e9)
            touched_sipms2.append(1.e9)
            true_r2       .append(1.e9)
            mean_phi2     .append(1.e9)
            mean_z2       .append(1.e9)
            charges2      .append(1.e9)

a_true_r1   = np.array(true_r1)
a_true_r2   = np.array(true_r2)
a_mean_phi1 = np.array(mean_phi1)
a_mean_phi2 = np.array(mean_phi2)
a_var_phi1  = np.array(var_phi1)
a_var_phi2  = np.array(var_phi2)
a_mean_z1   = np.array(mean_z1)
a_mean_z2   = np.array(mean_z2)
a_var_z1    = np.array(var_z1)
a_var_z2    = np.array(var_z2)

a_charges1 = np.array(charges1)
a_charges2 = np.array(charges2)

a_evt_ids       = np.array(evt_ids)
a_num_phot_evts = np.array(num_phot_evts)

a_touched_sipms1 = np.array(touched_sipms1)
a_touched_sipms2 = np.array(touched_sipms2)

np.savez(evt_file, true_r1=a_true_r1, true_r2=a_true_r2, 
var_phi1=a_var_phi1, var_phi2=a_var_phi2, 
var_z1=a_var_z1, var_z2=a_var_z2, 
charges1=a_charges1, charges2=a_charges2,
mean_phi1=a_mean_phi1, mean_phi2=a_mean_phi2, mean_z1=a_mean_z1, mean_z2=a_mean_z2,
evt_ids=a_evt_ids, num_phot_evts=a_num_phot_evts,
touched_sipms1=a_touched_sipms1, touched_sipms2=a_touched_sipms2)
