import sys
import numpy  as np
import pandas as pd

from invisible_cities.core                  import system_of_units as units
from invisible_cities.reco.sensor_functions import charge_fluctuation

import antea.database.load_db       as db
import antea.reco.reco_functions    as rf
import antea.reco.mctrue_functions  as mcf
import antea.elec.tof_functions     as tf
import antea.mcsim.sensor_functions as snsf

from antea.utils.map_functions import load_map
from antea.io.mc_io import read_sensor_bin_width_from_conf
from antea.io.mc_io import load_mcparticles, load_mchits
from antea.io.mc_io import load_mcsns_response, load_mcTOFsns_response
from antea.mcsim.sensor_functions import apply_charge_fluctuation

### read sensor positions from database
#DataSiPM     = db.DataSiPM('petalo', 0) # ring
DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
DataSiPM_idx = DataSiPM.set_index('SensorID')
n_sipms      = len(DataSiPM)
first_sipm   = DataSiPM_idx.index.min()

### parameters for single photoelectron convolution in SiPM response
tau_sipm       = [100, 15000]
time_window    = 5000
time           = np.arange(0, 5000)
spe_resp, norm = tf.apply_spe_dist(time, tau_sipm)


def reconstruct_position(q, pos, thr_r, thr_phi, thr_z):
    ## Calculate R
    posr, qr = rf.sel_coord(pos, q, thr_r)
    if len(posr) != 0:
        pos_phi = rf.from_cartesian_to_cyl(np.array(posr))[:,1]
        _, var_phi = rf.phi_mean_var(pos_phi, qr)
        r = Rpos(np.sqrt(var_phi)).value
    else:
        return 1.e9, 1.e9, 1.e9

    ## Calculate phi
    posphi, qphi = rf.sel_coord(pos, q, thr_phi)

    if len(qphi) != 0:
        reco_cart_pos = np.average(posphi, weights=qphi, axis=0)
        phi           = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])
    else:
        return 1.e9, 1.e9, 1.e9

    ## Calculate z
    posz, qz = rf.sel_coord(pos, q, thr_z)

    if len(qz) != 0:
        reco_cart_pos = np.average(posz, weights=qz, axis=0)
        z             = reco_cart_pos[2]
    else:
        return 1.e9, 1.e9, 1.e9

    return r, phi, z

def calculate_average_SiPM_pos(min_ids):
    sipm     = DataSiPM_idx.loc[np.abs(min_ids)]
    sipm_pos = np.array([sipm.X.values, sipm.Y.values, sipm.Z.values]).transpose()
    ave_pos  = np.average(sipm_pos, axis=0)

    return ave_pos


start   = int(sys.argv[1])
numb    = int(sys.argv[2])
thr_r   = 4
thr_phi = 4
thr_z   = 4
thr_e   = 2

n_pe       = 1
sigma_sipm = 40 #80 ps SiPM jitter
sigma_elec = 30 #30 ps electronic jitter

file_full = '/sim/full_body_phantom_paper/fullsim/full_body_phantom_paper.{0}.h5'
evt_file  = '/sim/analysis/jitter/full_body_phantom_paper_coincidences_jitter_sigSiPM_{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(sigma_sipm, start, numb, int(thr_r), int(thr_phi), int(thr_z), int(thr_e))
rpos_file = '/home/rolucar/Analysis/full_body_phantom_paper/r_table_full_body_phantom_paper_thr{0}pes.h5'.format(int(thr_r))

Rpos = load_map(rpos_file,
                group  = "Radius",
                node   = "f{}pes150bins".format(int(thr_r)), # 150, 100
                x_name = "PhiRms",
                y_name = "Rpos",
                u_name = "RposUncertainty")

c0 = c1 = 0
bad = 0

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
true_time1, true_time2               = [], []
touched_sipms1, touched_sipms2       = [], []
photo1, photo2                  = [], []
max_hit_distance1, max_hit_distance2 = [], []

event_ids = []


for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        sns_response = load_mcsns_response(file_name)
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} not found'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/sns_response in file {0}'.format(file_name))
        continue
    print('Analyzing file {0}'.format(file_name))

    #sns_response = snsf.apply_sipm_pde(sns_response, 0.3)
    fluct_sns_response = apply_charge_fluctuation(sns_response, DataSiPM_idx)

    tof_bin_size = read_sensor_bin_width_from_conf(file_name, tof=True)

    particles    = load_mcparticles      (file_name)
    hits         = load_mchits           (file_name)
    tof_response = load_mcTOFsns_response(file_name)
    #tof_response = snsf.apply_sipm_pde(tof_response, 0.3)

    events = particles.event_id.unique()
    print(f'Number of events = {len(events)}')
    charge_range = (2000, 2250) # range to select photopeak - to be adjusted to the specific case
#    charge_range = (2000, 3000) # range to select photopeak - to be adjusted to the specific case
#    charge_range = (1910, 2350) # range to select photopeak - to be adjusted to the specific case

    for evt in events:

        evt_sns = fluct_sns_response[fluct_sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=thr_e)
        if len(evt_sns) == 0:
            continue

        ids_over_thr = evt_sns.sensor_id.astype('int64').values

        evt_parts = particles   [particles   .event_id == evt]
        evt_hits  = hits        [hits        .event_id == evt]
        evt_tof   = tof_response[tof_response.event_id == evt]
        evt_tof   = evt_tof     [evt_tof     .sensor_id.isin(-ids_over_thr)]
        if len(evt_tof) == 0:
            continue

        pos1, pos2, q1, q2, true_pos1, true_pos2, true_t1, true_t2, sns1, sns2 = rf.reconstruct_coincidences(evt_sns, charge_range, DataSiPM_idx, evt_parts, evt_hits)
        if len(pos1) == 0 or len(pos2) == 0:
            c0 += 1
            continue

        q1   = np.array(q1)
        q2   = np.array(q2)
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        r1, phi1, z1 = reconstruct_position(q1, pos1, thr_r, thr_phi, thr_z)
        r2, phi2, z2 = reconstruct_position(q2, pos2, thr_r, thr_phi, thr_z)

        if (r1 > 1.e8) or (r2 > 1.e8):
            c1 += 1
            continue

        ## Use absolute times in units of ps
        times = evt_tof.time_bin.values * tof_bin_size / units.ps
        ## add SiPM jitter, if different from zero
        if sigma_sipm > 0:
            times = np.round(np.random.normal(times, sigma_sipm))
        evt_tof.insert(len(evt_tof.columns), 'time', times.astype(int)) # here we have bins of 1 ps

        ## produce a TOF dataframe with convolved time response
        tof_sns = evt_tof.sensor_id.unique()

        evt_tof_exp_dist = []
        for s_id in tof_sns:
            tdc_conv    = tf.tdc_convolution(evt_tof, spe_resp, s_id, time_window)
            tdc_conv_df = tf.translate_charge_conv_to_wf_df(evt, s_id, tdc_conv)
            if sigma_elec > 0:
                tdc_conv_df = tdc_conv_df.assign(time=np.random.normal(tdc_conv_df.time.values, sigma_elec))
            tdc_conv_df = tdc_conv_df[tdc_conv_df.charge > timestamp_thr/norm]
            tdc_conv_df = tdc_conv_df[tdc_conv_df.time == tdc_conv_df.time.min()]
            evt_tof_exp_dist.append(tdc_conv_df)
        evt_tof_exp_dist = pd.concat(evt_tof_exp_dist)

        try:
            min_id1, min_id2, min_t1, min_t2 = rf.find_coincidence_timestamps(evt_tof_exp_dist, sns1, sns2, n_pe)
            ave_pos1 = calculate_average_SiPM_pos(min_id1)
            ave_pos2 = calculate_average_SiPM_pos(min_id2)
            first_sipm1.append(ave_pos1)
            first_sipm2.append(ave_pos2)
        except WaveformEmptyTable:
            print(f'TOF dataframe has no minimum time for event {evt}')
            _, _, min_t1, min_t2 = [-1], [-1], -1, -1
            first_sipm1.append(np.array([0, 0, 0]))
            first_sipm2.append(np.array([0, 0, 0]))


        first_time1.append(min_t1)
        first_time2.append(min_t2)


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
        distances1 = rf.find_hit_distances_from_true_pos(evt_hits, true_pos1)
        max_dist1  = distances1.max()
        distances2 = rf.find_hit_distances_from_true_pos(evt_hits, true_pos2)
        max_dist2  = distances2.max()

        event_ids        .append(evt)
        reco_r1          .append(r1)
        reco_phi1        .append(phi1)
        reco_z1          .append(z1)
        true_r1          .append(np.sqrt(true_pos1[0]**2 + true_pos1[1]**2))
        true_phi1        .append(np.arctan2(true_pos1[1], true_pos1[0]))
        true_z1          .append(true_pos1[2])
        sns_response1    .append(sum(q1))
        touched_sipms1   .append(len(q1))
        true_time1       .append(true_t1/units.ps)
        photo1           .append(phot1)
        max_hit_distance1.append(max_dist1)
        reco_r2          .append(r2)
        reco_phi2        .append(phi2)
        reco_z2          .append(z2)
        true_r2          .append(np.sqrt(true_pos2[0]**2 + true_pos2[1]**2))
        true_phi2        .append(np.arctan2(true_pos2[1], true_pos2[0]))
        true_z2          .append(true_pos2[2])
        sns_response2    .append(sum(q2))
        touched_sipms2   .append(len(q2))
        true_time2       .append(true_t2/units.ps)
        photo2           .append(phot2)
        max_hit_distance2.append(max_dist2)


a_true_r1           = np.array(true_r1)
a_true_phi1         = np.array(true_phi1)
a_true_z1           = np.array(true_z1)
a_reco_r1           = np.array(reco_r1)
a_reco_phi1         = np.array(reco_phi1)
a_reco_z1           = np.array(reco_z1)
a_sns_response1     = np.array(sns_response1)
a_touched_sipms1    = np.array(touched_sipms1)
a_first_sipm1       = np.array(first_sipm1)
a_first_time1       = np.array(first_time1)
a_true_time1        = np.array(true_time1)
a_photo1            = np.array(photo1)
a_max_hit_distance1 = np.array(max_hit_distance1)

a_true_r2           = np.array(true_r2)
a_true_phi2         = np.array(true_phi2)
a_true_z2           = np.array(true_z2)
a_reco_r2           = np.array(reco_r2)
a_reco_phi2         = np.array(reco_phi2)
a_reco_z2           = np.array(reco_z2)
a_sns_response2     = np.array(sns_response2)
a_touched_sipms2    = np.array(touched_sipms2)
a_first_sipm2       = np.array(first_sipm2)
a_first_time2       = np.array(first_time2)
a_true_time2        = np.array(true_time2)
a_photo2            = np.array(photo2)
a_max_hit_distance2 = np.array(max_hit_distance2)

a_event_ids = np.array(event_ids)

np.savez(evt_file,
         a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1,
         a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
         a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
         a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
         a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2,
         a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2,
         a_first_sipm1=a_first_sipm1, a_first_time1=a_first_time1,
         a_first_sipm2=a_first_sipm2, a_first_time2=a_first_time2,
         a_true_time1=a_true_time1, a_true_time2=a_true_time2,
         a_photo1=a_photo1, a_photo2=a_photo2,
         a_max_hit_distance1=a_max_hit_distance1, a_max_hit_distance2=a_max_hit_distance2,
         a_event_ids=a_event_ids)

print(f'Not a coincidence: {c0}')
print(f'Not passing threshold to reconstruct position = {c1}')
