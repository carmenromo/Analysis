import sys
import tables as tb
import numpy as np

import optimization_utils as ots
import analysis_utils     as ats

from   antea.io.mc_io_tb    import read_SiPM_bin_width_from_conf
from   antea.io.mc_io_tb    import go_through_file
from   antea.utils.table_functions             import load_rpos

from invisible_cities.io.mcinfo_io  import read_mcinfo

start   = int(sys.argv[1])
numb    = int(sys.argv[2])

base_file = '/data/PETALO/pitch7mm/full_ring_iradius165mm_z140mm_depth3cm_pitch7mm.{0:03d}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/test_singles_{0}_{1}'.format(start, numb)

rpos_threshold = 4
rpos_file = '/home/paolafer/analysis/petalo/tables/r_table_iradius165mm_depth3cm_pitch7mm_new_h5_clean_thr4pes.h5'

Rpos = load_rpos(rpos_file,
                 group = "Radius",
                 node  = "f4pes200bins")


phi_threshold  = 5
zpos_threshold = 4
e_threshold    = 2

reco_r1, reco_r2, true_r1, true_r2          = [], [], [], []
reco_phi1, reco_phi2, true_phi1, true_phi2  = [], [], [], []
reco_z1, reco_z2, true_z1, true_z2          = [], [], [], []
events1, events2, sns_response1, sns_response2 = [], [], [], []

for number in range(start, start+numb):
    file_name = base_file.format(number)
    try:
        print('Trying file {0}'.format(file_name))
        h5in = tb.open_file(file_name, mode='r')
    except ValueError:
        continue
    except OSError:
         continue
    print('Analyzing file {0}'.format(file_name))

    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = ats.sensor_position    (h5in)
    sens_pos_cyl   = ats.sensor_position_cyl(h5in)
    bin_width      = read_SiPM_bin_width_from_conf(h5in)

    charge_range = (1000, 1400)
    single_events = []

    for evt in range(events_in_file):
        event_number   = h5in.root.MC.extents[evt]['evt_number']
        this_event_wvf = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')
        sns_over_thr, charges_over_thr = ats.find_SiPMs_over_threshold(this_event_wvf, e_threshold)
        if len(charges_over_thr) == 0: continue

        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        i1, i2, pos_true1, pos_true2, gamma1, gamma2, q1, q2, pos1, pos2 = ats.select_true_pos_from_charge(sns_over_thr, charges_over_thr, charge_range, sens_pos, part_dict)

        if i1 == i2: continue

        if i1:
            positions, qs = ots.reco_pos_single(pos_true1, np.array(q1), np.array(pos1), rpos_threshold, phi_threshold, zpos_threshold)
            gamma_sign = gamma1
            pos_true   = pos_true1

        else:
            positions, qs = ots.reco_pos_single(pos_true2, np.array(q2), np.array(pos2), rpos_threshold, phi_threshold, zpos_threshold)
            gamma_sign = gamma2
            pos_true   = pos_true2

        single_elem = (evt, gamma_sign, pos_true, positions, qs)
        single_events.append(single_elem)

    if len(single_events) % 2 == 1:
        del single_events[-1]

    for i in range(0, len(single_events), 2):
        ev1, sig1, pos_true1, sns_pos1, sns_qs1 = single_events[i]
        ev2, sig2, pos_true2, sns_pos2, sns_qs2 = single_events[i+1]
        p1    = ots.get_gamma_p(h5in, ev1, sig1)
        p2    = ots.get_gamma_p(h5in, ev2, sig2)
        theta = ots.get_theta(p1, p2)
        axis  = ots.get_axis (p1, p2)

        positions1, qs1 = ots.reco_pos_single(pos_true1, np.array(sns_qs1), np.array(sns_pos1), rpos_threshold, phi_threshold, zpos_threshold)
        positions2, qs2 = ots.reco_pos_single(pos_true1, np.array(sns_qs1), np.array(sns_pos1), rpos_threshold, phi_threshold, zpos_threshold)

        if len(positions1) == 0 or len(positions2) == 0:
            continue

        phi1        = ats.from_cartesian_to_cyl(positions1[0])[:,1]
        var_phi1    = ats.get_var_phi(phi1, qs1[0])
        sigma_phi1  = np.sqrt(var_phi1)
        reco1_r     = Rpos(sigma_phi1).value

        phi2        = ats.from_cartesian_to_cyl(positions2[0])[:,1]
        var_phi2    = ats.get_var_phi(phi2, qs2[0])
        sigma_phi2  = np.sqrt(var_phi2)
        reco2_r     = Rpos(sigma_phi2).value

        reco_cart = ats.barycenter_3D(positions1[1], qs1[1])
        reco1_phi = np.arctan2(reco_cart[1], reco_cart[0])

        reco_cart = ats.barycenter_3D(positions2[1], qs2[1])
        reco2_phi = np.arctan2(reco_cart[1], reco_cart[0])

        reco_cart = ats.barycenter_3D(positions1[2], qs1[2])
        reco1_z   = reco_cart[2]

        reco_cart = ats.barycenter_3D(positions2[2], qs2[2])
        reco2_z   = reco_cart[2]

        true1_r   = ats.from_cartesian_to_cyl(np.array([pos_true1]))[0, 0]
        true1_phi = ats.from_cartesian_to_cyl(np.array([pos_true1]))[0, 1]
        true1_z   = ats.from_cartesian_to_cyl(np.array([pos_true1]))[0, 2]

        true2_r   = ats.from_cartesian_to_cyl(np.array([pos_true2]))[0, 0]
        true2_phi = ats.from_cartesian_to_cyl(np.array([pos_true2]))[0, 1]
        true2_z   = ats.from_cartesian_to_cyl(np.array([pos_true2]))[0, 2]

        event_number1 = h5in.root.MC.extents[ev1]['evt_number']
        event_number2 = h5in.root.MC.extents[ev2]['evt_number']

        reco_r1.append(reco1_r)
        reco_phi1.append(reco1_phi)
        reco_z1.append(reco1_z)
        reco_r2.append(reco2_r)
        reco_phi2.append(reco2_phi)
        reco_z2.append(reco2_z)
        true_r1.append(true1_r)
        true_phi1.append(true1_phi)
        true_z1.append(true1_z)
        true_r2.append(true2_r)
        true_phi2.append(true2_phi)
        true_z2.append(true2_z)

        sns_response1.append(sum(sns_qs1))
        sns_response2.append(sum(sns_qs2))

        events1.append(event_number1)
        events2.append(event_number2)


a_true_r1    = np.array(true_r1)
a_true_phi1  = np.array(true_phi1)
a_true_z1    = np.array(true_z1)
a_reco_r1    = np.array(reco_r1)
a_reco_phi1  = np.array(reco_phi1)
a_reco_z1    = np.array(reco_z1)
a_sns_response1 = np.array(sns_response1)

a_true_r2    = np.array(true_r2)
a_true_phi2  = np.array(true_phi2)
a_true_z2    = np.array(true_z2)
a_reco_r2    = np.array(reco_r2)
a_reco_phi2  = np.array(reco_phi2)
a_reco_z2    = np.array(reco_z2)
a_sns_response2 = np.array(sns_response2)

a_events1 = np.array(events1)
a_events2 = np.array(events2)

np.savez(evt_file,    a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1,
                      a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
                      a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
                      a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
                      a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2,
                      a_events1=a_events1, a_events2=a_events2)
