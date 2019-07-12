import sys

import optimization_utils as ots
import analysis_utils     as ats

from   antea.io.mc_io_tb    import read_SiPM_bin_width_from_conf
from   antea.io.mc_io_tb    import go_through_file
from   antea.utils.table_functions             import load_rpos

start   = int(sys.argv[1])
numb    = int(sys.argv[2])

base_file = '/data/PETALO/pitch7mm/full_ring_iradius165mm_z140mm_depth3cm_pitch7mm.{0:03d}.pet.h5'
evt_file = '/home/paolafer/analysis/petalo/full_ring_iradius165mm_z140mm_depth3cm_pitch7mm_reco_pos_dnn_compt_compare_{0}_{1}_{2}'.format(start, numb, max_cut)

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
events, sns_response1, sns_response2 = [], [], []

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

    for evt in range(events_in_file):
        event_number = h5in.root.MC.extents[evt]['evt_number']
        this_event_wvf = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')
        sns_over_thr, charges_over_thr = ats.find_SiPMs_over_threshold(this_event_wvf, e_threshold)
        if len(charges_over_thr) == 0: continue

        i1, i2, pos_true1, pos_true2, q1, q2, pos1, pos2 = ats.select_true_pos_from_charge(sns_over_thr, charges_over_thr, charge_range, sens_pos)

        if i1 and i2:
            positions1, qs1 = ots.reco_pos_single(pos_true1, q1, pos1, rpos_threshold, th_phi_threshold, zpos_threshold)
            positions2, qs2 = ots.reco_pos_single(pos_true2, q2, pos2, rpos_threshold, th_phi_threshold, zpos_threshold)

            if len(positions1) == 0 or len(positions2) == 0:
                continue

            phi1        = ats.from_cartesian_to_cyl(positions1[0])[:,1]
            _, var_phi1 = ats.get_r_and_var_phi(pos_true1, phi1, qs1[0])
            sigma_phi1  = np.sqrt(var_phi1)
            reco1_r     = Rpos(sigma_phi1).value

            phi2        = ats.from_cartesian_to_cyl(positions2[0])[:,1]
            _, var_phi2 = ats.get_r_and_var_phi(pos_true2, phi2, qs2[0])
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

            true1_r   = ats.from_cartesian_to_cyl([pos_true1])[0, 0]
            true1_phi = ats.from_cartesian_to_cyl([pos_true1])[0, 1]
            true1_z   = ats.from_cartesian_to_cyl([pos_true1])[0, 2]

            true2_r   = ats.from_cartesian_to_cyl([pos_true2])[0, 0]
            true2_phi = ats.from_cartesian_to_cyl([pos_true2])[0, 1]
            true2_z   = ats.from_cartesian_to_cyl([pos_true2])[0, 2]

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

            sns_response1.append(q1)
            sns_response2.append(q2)

            events.append(event_number)


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

a_events = np.array(events)

np.savez(evt_file,    a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1,
                      a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
                      a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
                      a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
                      a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2,
                      a_events=a_events)
            
            

