import sys
import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.core         import system_of_units as units
from invisible_cities.io.mcinfo_io import units_dict

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.utils.table_functions import load_rpos
from antea.io.mc_io import read_sensor_bin_width_from_conf


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

start   = int(sys.argv[1])
numb    = int(sys.argv[2])
thr_r   = float(sys.argv[3])
thr_phi = float(sys.argv[4])
thr_z   = float(sys.argv[5])
thr_e   = float(sys.argv[6])

folder = '/data/PETALO/full_body/phantom/'
file_full = folder + 'full_body_phantom.{0:03d}.pet.h5'
evt_file  = '/home/paolafer/analysis/petalo/phantom/full_body_phantom_coincidences_phot_info_{0}_{1}_{2}_{3}_{4}_{5}'.format(start, numb, int(thr_r), int(thr_phi), int(thr_z), int(thr_e))

rpos_file = '/home/paolafer/analysis/petalo/tables/r_table_full_body_195cm_thr{}pes.h5'.format(int(thr_r))

Rpos = load_rpos(rpos_file,
                 group = "Radius",
                 node  = "f{}pes200bins".format(int(thr_r)))

c0 = c1 = c2 = c3 = c4 = 0
bad = 0

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

sns_response1, sns_response2    = [], []
first_sipm1, first_sipm2        = [], []
first_time1, first_time2        = [], []
true_time1, true_time2          = [], []
touched_sipms1, touched_sipms2  = [], []
photo1, photo_like1           = [], []
photo2, photo_like2           = [], []
hit_energy1, hit_energy2        = [], []

event_ids = []


for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        sns_response = pd.read_hdf(file_name, 'MC/waveforms')
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} not found'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/waveforms in file {0}'.format(file_name))
        continue
    print('Analyzing file {0} new script'.format(file_name))

    h5f = tb.open_file(file_name, mode='r')
    tof_bin_size = read_sensor_bin_width_from_conf(h5f, tof=True)
    h5f.close()

    particles = pd.read_hdf(file_name, 'MC/particles')
    hits      = pd.read_hdf(file_name, 'MC/hits')
    tof_response = pd.read_hdf(file_name, 'MC/tof_waveforms')

    events = particles.event_id.unique()
    print(len(events))
    charge_range = (1050, 1300)

    for evt in events[:]:

        evt_sns = sns_response[sns_response.event_id == evt]
        evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=2)
        if len(evt_sns) == 0:
            continue

        evt_parts = particles[particles.event_id       == evt]
        evt_hits  = hits[hits.event_id                 == evt]
        evt_tof   = tof_response[tof_response.event_id == evt]

        pos1, pos2, q1, q2, true_pos1, true_pos2, true_t1, true_t2, min_id1, min_id2, min_t1, min_t2 = rf.reconstruct_coincidences(evt_sns, evt_tof, charge_range, DataSiPM_idx, evt_parts, evt_hits)
        if len(pos1) == 0 or len(pos2) == 0:
            c0 += 1
            continue

        q1   = np.array(q1);
        q2   = np.array(q2);
        pos1 = np.array(pos1);
        pos2 = np.array(pos2);

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

        event_ids.append(evt)
        reco_r1.append(r1)
        reco_phi1.append(phi1)
        reco_z1.append(z1)
        true_r1.append(np.sqrt(true_pos1[0]**2 + true_pos1[1]**2))
        true_phi1.append(np.arctan2(true_pos1[1], true_pos1[0]))
        true_z1.append(true_pos1[2])
        sns_response1.append(sum(q1e))
        touched_sipms1.append(len(q1e))
        first_sipm1.append(min_id1)
        first_time1.append(min_t1*tof_bin_size/units.ps)
        true_time1.append(true_t1/units.ps)
        photo1.append(phot1)
        photo_like1.append(phot_like1)
        hit_energy1.append(tot_hit_energy1)

        reco_r2.append(r2)
        reco_phi2.append(phi2)
        reco_z2.append(z2)
        true_r2.append(np.sqrt(true_pos2[0]**2 + true_pos2[1]**2))
        true_phi2.append(np.arctan2(true_pos2[1], true_pos2[0]))
        true_z2.append(true_pos2[2])
        sns_response2.append(sum(q2e))
        touched_sipms2.append(len(q2e))
        first_sipm2.append(min_id2)
        first_time2.append(min_t2*tof_bin_size/units.ps)
        true_time2.append(true_t2/units.ps)
        photo2.append(phot2)
        photo_like2.append(phot_like2)
        hit_energy2.append(tot_hit_energy2)


a_true_r1   = np.array(true_r1)
a_true_phi1 = np.array(true_phi1)
a_true_z1   = np.array(true_z1)
a_reco_r1   = np.array(reco_r1)
a_reco_phi1 = np.array(reco_phi1)
a_reco_z1   = np.array(reco_z1)
a_sns_response1  = np.array(sns_response1)
a_touched_sipms1 = np.array(touched_sipms1)
a_first_sipm1 = np.array(first_sipm1)
a_first_time1 = np.array(first_time1)
a_true_time1  = np.array(true_time1)
a_photo1    = np.array(photo1)
a_photo_like1    = np.array(photo_like1)
a_hit_energy1 = np.array(hit_energy1)

a_true_r2   = np.array(true_r2)
a_true_phi2 = np.array(true_phi2)
a_true_z2   = np.array(true_z2)
a_reco_r2   = np.array(reco_r2)
a_reco_phi2 = np.array(reco_phi2)
a_reco_z2   = np.array(reco_z2)
a_sns_response2  = np.array(sns_response2)
a_touched_sipms2 = np.array(touched_sipms2)
a_first_sipm2 = np.array(first_sipm2)
a_first_time2 = np.array(first_time2)
a_true_time2  = np.array(true_time2)
a_photo2    = np.array(photo2)
a_photo_like2    = np.array(photo_like2)
a_hit_energy2 = np.array(hit_energy2)

a_event_ids = np.array(event_ids)

np.savez(evt_file, a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1, a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2, a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1, a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2, a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2, a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2, a_first_sipm1=a_first_sipm1, a_first_time1=a_first_time1, a_first_sipm2=a_first_sipm2, a_first_time2=a_first_time2, a_true_time1=a_true_time1, a_true_time2=a_true_time2, a_photo1=a_photo1, a_photo2=a_photo2, a_photo_like1=a_photo_like1, a_photo_like2=a_photo_like2, a_hit_energy1=a_hit_energy1, a_hit_energy2=a_hit_energy2, a_event_ids=a_event_ids)

print('Not a coincidence: {}'.format(c0))
print('Not passing threshold r = {}, phi = {}, z = {}, E = {}'.format(c1, c2, c3, c4))




