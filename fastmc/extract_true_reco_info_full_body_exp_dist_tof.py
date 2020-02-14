import sys
import argparse
import numpy    as np
import pandas   as pd
import tables   as tb

from invisible_cities.core         import system_of_units as units

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf
import antea.elec.tof_functions    as tf

from antea.utils.table_functions import load_rpos
from antea.io   .mc_io           import load_mchits
from antea.io   .mc_io           import load_mcparticles
from antea.io   .mc_io           import load_mcsns_response
from antea.io   .mc_io           import load_mcTOFsns_response
from antea.io.mc_io import read_sensor_bin_width_from_conf


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')


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
    parser.add_argument('base_path'  ,             help = "Path of the conf"           )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()


arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
thr_r      = arguments.thr_r
thr_phi    = arguments.thr_phi
thr_z      = arguments.thr_z
thr_e      = arguments.thr_e
eventsPath = arguments.events_path
file_name  = arguments.file_name
base_path  = arguments.base_path
data_path  = arguments.data_path

evt_file  = f"{base_path}/{data_path}/full_body_phantom_coincidences_phot_info_dist_tof_{start}_{numb}_{thr_r}_{thr_phi}_{thr_z}_{thr_e}"
rpos_file = f"r_table_full_body_195cm_thr{thr_r}pes.h5"

Rpos = load_rpos(rpos_file,
                 group = "Radius",
                 node  = f"f{thr_r}pes200bins")

### TOF elec parameters:
te_tdc         = 0.25
n_sipms        = len(DataSiPM)
first_sipm     = DataSiPM_idx.index.min()
time_window    = 10000
time_bin       = 5
time           = np.arange(0, 80000, time_bin)
spe_resp, norm = tf.apply_spe_dist(time)


c0 = c1 = c2 = c3 = c4 = 0
bad = 0

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

sns_response1, sns_response2   = [], []
first_sipm1, first_sipm2       = [], []
first_time1, first_time2       = [], []
true_time1, true_time2         = [], []
touched_sipms1, touched_sipms2 = [], []
photo1, photo_like1            = [], []
photo2, photo_like2            = [], []
hit_energy1, hit_energy2       = [], []

event_ids = []


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

    charge_range = (1050, 1300)

    for evt in events[:]:
        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=2)
        if len(evt_sns) == 0:
            continue

        evt_parts = particles       [particles       .event_id == evt]
        evt_hits  = hits            [hits            .event_id == evt]
        evt_tof   = sns_response_tof[sns_response_tof.event_id == evt]

        pos1, pos2, q1, q2, true_pos1, true_pos2, true_t1, true_t2 = rf.select_coincidences(evt_sns, evt_tof, charge_range, DataSiPM_idx, evt_parts, evt_hits)

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
        q1r    = q1  [sel1_r]
        pos1r  = pos1[sel1_r]
        sel2_r = q2>thr_r
        q2r    = q2  [sel2_r]
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

        ## extract information about the interaction being photoelectric
        phot, phot_pos = mcf.select_photoelectric(evt_parts, evt_hits)
        if not phot:
            phot1 = False
            phot2 = False
        else:
            scalar_prod = true_pos1.dot(phot_pos[0])
            if scalar_prod > 0:
                phot1 = True
                phot2 = False
            else:
                phot1 = False
                phot2 = True

            if len(phot_pos) == 2:
                if scalar_prod > 0:
                    phot2 = True
                else:
                    phot1 = True

        ## extract information about the interaction being photoelectric-like
        positions         = np.array([evt_hits.x, evt_hits.y, evt_hits.z]).transpose()
        scalar_products1 = positions.dot(true_pos1)
        hits1 = evt_hits[scalar_products1 >= 0]
        pos_hits1  = np.array([hits1.x, hits1.y, hits1.z]).transpose()
        distances1 = np.linalg.norm(np.subtract(pos_hits1, true_pos1), axis=1)
        if len(distances1[distances1 > 1.]): ## hits at <1 mm distance are considered of the same point
            phot_like1 = False
        else:
            phot_like1 = True

        hits2 = evt_hits[scalar_products1 < 0]
        pos_hits2  = np.array([hits2.x, hits2.y, hits2.z]).transpose()
        distances2 = np.linalg.norm(np.subtract(pos_hits2, true_pos2), axis=1)

        if len(distances2[distances2 > 1.]):
            phot_like2 = False
        else:
            phot_like2 = True

        tot_hit_energy1 = hits1.energy.sum()
        tot_hit_energy2 = hits2.energy.sum()


        ## Tof convolution
        tof_sns = evt_tof.sensor_id.unique()
        evt_tof_exp_dist = []
        for s_id in tof_sns[:]:
            tdc_conv    = tf.tdc_convolution(evt_tof, spe_resp, time_window, s_id, te_tdc)
            tdc_conv_df = tf.translate_charge_conv_to_wf_df(evt, s_id, tdc_conv)
            evt_tof_exp_dist.append(tdc_conv_df)
        evt_tof_exp_dist = pd.concat(evt_tof_exp_dist)

        min_id1, min_id2, min_t1, min_t2 = rf.find_first_times_of_coincidences(evt_sns, evt_tof_exp_dist, charge_range, DataSiPM_idx, evt_parts, evt_hits)


        event_ids     .append(evt)
        reco_r1       .append(r1)
        reco_phi1     .append(phi1)
        reco_z1       .append(z1)
        true_r1       .append(np.sqrt(true_pos1[0]**2 + true_pos1[1]**2))
        true_phi1     .append(np.arctan2(true_pos1[1], true_pos1[0]))
        true_z1       .append(true_pos1[2])
        sns_response1 .append(sum(q1e))
        touched_sipms1.append(len(q1e))
        first_sipm1   .append(min_id1)
        first_time1   .append(min_t1*tof_bin_size/units.ps)
        true_time1    .append(true_t1/units.ps)
        photo1        .append(phot1)
        photo_like1   .append(phot_like1)
        hit_energy1   .append(tot_hit_energy1)

        reco_r2       .append(r2)
        reco_phi2     .append(phi2)
        reco_z2       .append(z2)
        true_r2       .append(np.sqrt(true_pos2[0]**2 + true_pos2[1]**2))
        true_phi2     .append(np.arctan2(true_pos2[1], true_pos2[0]))
        true_z2       .append(true_pos2[2])
        sns_response2 .append(sum(q2e))
        touched_sipms2.append(len(q2e))
        first_sipm2   .append(min_id2)
        first_time2   .append(min_t2*tof_bin_size/units.ps)
        true_time2    .append(true_t2/units.ps)
        photo2        .append(phot2)
        photo_like2   .append(phot_like2)
        hit_energy2   .append(tot_hit_energy2)


a_true_r1        = np.array(true_r1)
a_true_phi1      = np.array(true_phi1)
a_true_z1        = np.array(true_z1)
a_reco_r1        = np.array(reco_r1)
a_reco_phi1      = np.array(reco_phi1)
a_reco_z1        = np.array(reco_z1)
a_sns_response1  = np.array(sns_response1)
a_touched_sipms1 = np.array(touched_sipms1)
a_first_sipm1    = np.array(first_sipm1)
a_first_time1    = np.array(first_time1)
a_true_time1     = np.array(true_time1)
a_photo1         = np.array(photo1)
a_photo_like1    = np.array(photo_like1)
a_hit_energy1    = np.array(hit_energy1)

a_true_r2        = np.array(true_r2)
a_true_phi2      = np.array(true_phi2)
a_true_z2        = np.array(true_z2)
a_reco_r2        = np.array(reco_r2)
a_reco_phi2      = np.array(reco_phi2)
a_reco_z2        = np.array(reco_z2)
a_sns_response2  = np.array(sns_response2)
a_touched_sipms2 = np.array(touched_sipms2)
a_first_sipm2    = np.array(first_sipm2)
a_first_time2    = np.array(first_time2)
a_true_time2     = np.array(true_time2)
a_photo2         = np.array(photo2)
a_photo_like2    = np.array(photo_like2)
a_hit_energy2    = np.array(hit_energy2)

a_event_ids      = np.array(event_ids)

np.savez(evt_file, a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1, a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
                   a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1, a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
                   a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2, a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2,
                   a_first_sipm1=a_first_sipm1, a_first_time1=a_first_time1, a_first_sipm2=a_first_sipm2, a_first_time2=a_first_time2, a_true_time1=a_true_time1,
                   a_true_time2=a_true_time2, a_photo1=a_photo1, a_photo2=a_photo2, a_photo_like1=a_photo_like1, a_photo_like2=a_photo_like2, a_hit_energy1=a_hit_energy1, a_hit_energy2=a_hit_energy2, a_event_ids=a_event_ids)

print(f"Not a coincidence: {c0}")
print(f"Not passing threshold r = {c1}, phi = {c2}, z = {c3}, E = {c4}")




