import sys
import argparse
import numpy  as np
import pandas as pd
import tables as tb

import pdb

from invisible_cities.core         import system_of_units as units

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import antea.elec.tof_functions as tf
import antea.mcsim.sensor_functions as snsf

from antea.utils.map_functions import load_map
from antea.io.mc_io import read_sensor_bin_width_from_conf

from antea.core.exceptions import WaveformEmptyTable


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
    parser.add_argument('rpos_file'  ,             help = "File of the Rpos"           )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')
n_sipms        = len(DataSiPM)
first_sipm     = DataSiPM_idx.index.min()


### parameters for single photoelectron convolution in SiPM response
tau_sipm       = [100, 15000]
time_window    = 5000
#time_bin       = 5 # ps
time           = np.arange(0, 5000)
#time           = time + (time_bin/2)
spe_resp, norm = tf.apply_spe_dist(time, tau_sipm)


sigma_sipm = 0 #80 #ps
sigma_elec = 0 #30 #ps
n_pe = 10

arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
thr_r      = arguments.thr_r
thr_phi    = arguments.thr_phi
thr_z      = arguments.thr_z
thr_e      = arguments.thr_e
eventsPath = arguments.events_path
file_name  = arguments.file_name
rpos_file  = arguments.rpos_file
data_path  = arguments.data_path

print(f'Using r map: {rpos_file}')

evt_file  = f"{data_path}/tof_coincidences_Paola_npe_{n_pe}_{start}_{numb}_{thr_r}_{thr_phi}_{thr_z}_{thr_e}"
Rpos = load_map(rpos_file,
                group  = "Radius",
                node   = f"f{int(thr_r)}pes150bins",
                x_name = "PhiRms",
                y_name = "Rpos",
                u_name = "RposUncertainty")

#charge_range = (2000, 2250) # pde 0.30, n=1.6
charge_range = (0, 5000)
print(f'Charge range = {charge_range}')
c0 = c1 = c2 = c3 = c4 = 0
bad = 0
boh0 = boh1 = 0
below_thr = 0

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

sns_response1, sns_response2    = [], []

### PETsys thresholds to extract the timestamp
timestamp_thr = 0.25
first_sipm1 = []
first_sipm2 = []
first_time1 = []
first_time2 = []
true_time1, true_time2          = [], []
touched_sipms1, touched_sipms2  = [], []
photo1, photo2 = [], []
max_hit_distance1, max_hit_distance2 = [], []
hit_energy1, hit_energy2        = [], []

event_ids = []


for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    try:
        #sns_response = load_mcsns_response(filename)
        sns_response = pd.read_hdf(filename, 'MC/waveforms')
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

    tof_bin_size = read_sensor_bin_width_from_conf(filename, tof=True)


    particles = pd.read_hdf(filename, 'MC/particles')
    hits      = pd.read_hdf(filename, 'MC/hits')
    sns_response = snsf.apply_sipm_pde(sns_response, 0.3)
    sns_response = snsf.apply_charge_fluctuation(sns_response, DataSiPM_idx)

    tof_response = pd.read_hdf(filename, 'MC/tof_waveforms')

    events = particles.event_id.unique()

    for evt in events[:]:

        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=thr_e)
        if len(evt_sns) == 0:
            boh0 += 1
            continue

        ids_over_thr = evt_sns.sensor_id.astype('int64').values

        evt_parts = particles[particles.event_id       == evt]
        evt_hits  = hits[hits.event_id                 == evt]
        evt_tof   = tof_response[tof_response.event_id == evt]

 #       if evt_hits.energy.sum() < 0.511:
 #           below_thr += 1
 #           continue
        if len(evt_tof) == 0:
            boh1 += 1
            continue
        evt_tof   = evt_tof[evt_tof.sensor_id.isin(-ids_over_thr)]
        if len(evt_tof) == 0:
            boh2 += 1
            continue

        pos1, pos2, q1, q2, true_pos1, true_pos2, true_t1, true_t2, sns1, sns2 = rf.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)
        if len(pos1) == 0 or len(pos2) == 0:
            c0 += 1
            continue

        q1   = np.array(q1)
        q2   = np.array(q2)
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        ## Calculate R
        r1 = r2 = None

        sel1_r = q1>thr_r
        q1r    = q1[sel1_r]
        pos1r  = pos1[sel1_r]
        sel2_r = q2>thr_r
        q2r    = q2[sel2_r]
        pos2r  = pos2[sel2_r]
        if len(pos1r) == 0 or len(pos2r) == 0:
            c1 += 1
            continue

        pos1_phi = rf.from_cartesian_to_cyl(np.array(pos1r))[:,1]
        diff_sign = min(pos1_phi ) < 0 < max(pos1_phi)
        if diff_sign & (np.abs(np.min(pos1_phi))>np.pi/2.):
            pos1_phi[pos1_phi<0] = np.pi + np.pi + pos1_phi[pos1_phi<0]
        mean_phi = np.average(pos1_phi, weights=q1r)
        var_phi1 = np.average((pos1_phi-mean_phi)**2, weights=q1r)
        r1  = Rpos(np.sqrt(var_phi1)).value

        pos2_phi = rf.from_cartesian_to_cyl(np.array(pos2r))[:,1]
        diff_sign = min(pos2_phi ) < 0 < max(pos2_phi)
        if diff_sign & (np.abs(np.min(pos2_phi))>np.pi/2.):
            pos2_phi[pos2_phi<0] = np.pi + np.pi + pos2_phi[pos2_phi<0]
        mean_phi = np.average(pos2_phi, weights=q2r)
        var_phi2 = np.average((pos2_phi-mean_phi)**2, weights=q2r)
        r2  = Rpos(np.sqrt(var_phi2)).value

        sel1_phi = q1>thr_phi
        q1phi    = q1[sel1_phi]
        pos1phi  = pos1[sel1_phi]
        sel2_phi = q2>thr_phi
        q2phi    = q2[sel2_phi]
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
        q1z    = q1[sel1_z]
        pos1z  = pos1[sel1_z]
        sel2_z = q2>thr_z
        q2z    = q2[sel2_z]
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

        times = evt_tof.time_bin.values * tof_bin_size / units.ps
        if sigma_sipm != 0:
            evt_tof.insert(len(evt_tof.columns), 'time', np.round(np.random.normal(times, sigma_sipm)).astype(int))
        else:
            evt_tof.insert(len(evt_tof.columns), 'time', times.astype(int))

        #print(evt_tof)
        ## produce a TOF dataframe with convolved time response
        tof_sns = evt_tof.sensor_id.unique()
        tof_exp = []
        for s_id in tof_sns:
            tdc_conv    = tf.tdc_convolution(evt_tof, spe_resp, s_id, time_window)
            tdc_conv_df = tf.translate_charge_conv_to_wf_df(evt, s_id, tdc_conv)
            if sigma_elec != 0:
                tdc_conv_df.assign(time=np.random.normal(tdc_conv_df.time.values, sigma_elec))

            tdc_conv_df = tdc_conv_df[tdc_conv_df.charge > timestamp_thr/norm]
            tdc_conv_df = tdc_conv_df[tdc_conv_df.time == tdc_conv_df.time.min()]
            #print(tdc_conv_df)
            tof_exp.append(tdc_conv_df)
        #pdb.set_trace()
        tof_exp = pd.concat(tof_exp)

        try:
            min_id1, min_id2, min_t1, min_t2 = rf.find_coincidence_timestamps(tof_exp, sns1, sns2, n_pe)
        except WaveformEmptyTable:
            continue


        sipm        = DataSiPM_idx.loc[np.abs(min_id1)]
        sipm_pos    = np.array([sipm.X.values, sipm.Y.values, sipm.Z.values]).transpose()
        ave_pos1    = np.average(sipm_pos, axis=0)

        sipm        = DataSiPM_idx.loc[np.abs(min_id2)]
        sipm_pos    = np.array([sipm.X.values, sipm.Y.values, sipm.Z.values]).transpose()
        ave_pos2    = np.average(sipm_pos, axis=0)

        first_sipm1.append(ave_pos1)
        first_time1.append(min_t1)

        first_sipm2.append(ave_pos2)
        first_time2.append(min_t2)



        ## extract information about the interaction being photoelectric-like
        positions         = np.array([evt_hits.x, evt_hits.y, evt_hits.z]).transpose()
        scalar_products1 = positions.dot(true_pos1)
        hits1 = evt_hits[scalar_products1 >= 0]
        pos_hits1  = np.array([hits1.x, hits1.y, hits1.z]).transpose()
        distances1 = np.linalg.norm(np.subtract(pos_hits1, true_pos1), axis=1)
        max_dist1  = distances1.max()

        hits2 = evt_hits[scalar_products1 < 0]
        pos_hits2  = np.array([hits2.x, hits2.y, hits2.z]).transpose()
        distances2 = np.linalg.norm(np.subtract(pos_hits2, true_pos2), axis=1)
        max_dist2  = distances2.max()

        event_ids.append(evt)
        true_time1.append(true_t1/units.ps)
        max_hit_distance1.append(max_dist1)
        true_time2.append(true_t2/units.ps)
        max_hit_distance2.append(max_dist2)

        reco_r1.append(r1)
        reco_phi1.append(phi1)
        reco_z1.append(z1)
        reco_r2.append(r2)
        reco_phi2.append(phi2)
        reco_z2.append(z2)


a_first_sipm1_1 = np.array(first_sipm1)
a_first_time1_1 = np.array(first_time1)
a_true_time1  = np.array(true_time1)
a_max_hit_distance1 = np.array(max_hit_distance1)
a_first_sipm2_1 = np.array(first_sipm2)
a_first_time2_1 = np.array(first_time2)
a_true_time2  = np.array(true_time2)
a_max_hit_distance2 = np.array(max_hit_distance2)

a_reco_r1   = np.array(reco_r1)
a_reco_phi1 = np.array(reco_phi1)
a_reco_z1   = np.array(reco_z1)
a_reco_r2   = np.array(reco_r2)
a_reco_phi2 = np.array(reco_phi2)
a_reco_z2   = np.array(reco_z2)

a_event_ids = np.array(event_ids)

np.savez(evt_file,
        a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
        a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
        a_first_sipm1_1=a_first_sipm1_1, a_first_time1_1=a_first_time1_1,
        a_first_sipm2_1=a_first_sipm2_1, a_first_time2_1=a_first_time2_1,
        a_true_time1=a_true_time1, a_true_time2=a_true_time2,
        a_max_hit_distance1=a_max_hit_distance1, a_max_hit_distance2=a_max_hit_distance2,
        a_event_ids=a_event_ids)

print('Not passing charge threshold = {}'.format(boh0))
print('Not passing tof charge threshold = {}'.format(boh1))
print('Not a coincidence: {}'.format(c0))
print(f'Number of coincidences: {len(a_event_ids)}')
print('Not passing threshold r = {}, phi = {}, z = {}, E = {}'.format(c1, c2, c3, c4))
print('Events below true energy threshold = {}'.format(below_thr))
