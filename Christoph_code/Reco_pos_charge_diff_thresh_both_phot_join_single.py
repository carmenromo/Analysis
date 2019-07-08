import os
import sys
import math
import datetime
import tables             as tb
import numpy              as np
import optimization_utils as ots
import pdb
import analysis_utils  as ats

from   antea.io   .mc_io                import read_mcsns_response
from   invisible_cities.io  .mcinfo_io  import read_mcinfo

from invisible_cities.core.exceptions   import SipmEmptyList
from invisible_cities.core.exceptions   import SipmZeroCharge

print(datetime.datetime.now())


start   = int(sys.argv[1])
numb    = int(sys.argv[2])

eventsPath = '/data4/PETALO/PETit-ring/7mm_pitch'
file_name  = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm'
base_path  = '/data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton'
data_path  = '/optimization_singles/3_data_reco_charge_diff_thr_both_phot'
evt_file    = '{0}/full_ring_irad15cm_d3cm_p7mm_pos_and_charge_difth_both_phot_photopeak_join_single_{1}_{2}'.format(base_path+data_path, start, numb)
rpos_file   = base_path+'/r_sigma_phi_table_iradius165mm_thr4pes_depth3cm_compton_sel_photopeak_new.h5'
Rpos        = ats.load_rpos(rpos_file, group = "Radius", node  = "f4pes150bins")

true_r1    = []
true_r2    = []
true_phi1  = []
true_phi2  = []
true_z1    = []
true_z2    = []
true_x1    = []
true_x2    = []
true_y1    = []
true_y2    = []

reco_r1    = []
reco_r2    = []
reco_phi1  = []
reco_phi2  = []
reco_z1    = []
reco_z2    = []
reco_x1    = []
reco_x2    = []
reco_y1    = []
reco_y2    = []

charge1    = []
charge2    = []
events1    = []
events2    = []
tot_charge = []
tot_evts   = 0

rpos_threshold = 3
phi_threshold  = 5
zpos_threshold = 4
e_threshold    = 2

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = '%s/%s.%s.pet.h5'%(eventsPath, file_name, number_str)

    print('Analyzing file {0}'.format(true_file))

    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        continue

    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = ats.sensor_position    (h5in)
    sens_pos_cyl   = ats.sensor_position_cyl(h5in)
    
    single_events = []

    for evt in range(events_in_file):
        #for evt in range(100):
        tot_evts += 1
        try:
            this_event_dict = read_mcinfo(h5in, (evt, evt+1))
            this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
            part_dict       = list(this_event_dict.values())[0]

            i1, i2, ave_true1, ave_true2 = ats.true_photoelect_compton(h5in, true_file, evt)

            if i1 == i2: 
                continue

            sns_dict    = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
            sns_ids     = np.array(list(sns_dict.keys()))

            indices_over_thr = (tot_charges > rpos_threshold)
            sns_over_thr     = sns_ids    [indices_over_thr]
            charges_over_thr = tot_charges[indices_over_thr]

            if len(charges_over_thr) == 0:
                continue

            sig_pgamma, ampl, ave_true = ots.single_sensor_classif(i1, i2, 
                                                                   ave_true1, 
                                                                   ave_true2, 
                                                                   sens_pos, 
                                                                   sens_pos_cyl, 
                                                                   sns_over_thr, 
                                                                   charges_over_thr)

            if ampl > 1000:
                sing_elem = (evt, sig_pgamma, ave_true)
                single_events.append(sing_elem)
            
        except tb.exceptions.HDF5ExtError:
            continue
        except ValueError:
            continue
        except OSError:
            continue
        except SipmEmptyList:
            continue

    if len(single_events) % 2 == 1:
        del single_events[-1]

    for i in range(0, len(single_events), 2):
        ev1, sig1, ave_true1 = single_events[i]
        ev2, sig2, ave_true2 = single_events[i+1]
        p1    = ots.get_gamma_p(h5in, ev1, sig1)
        p2    = ots.get_gamma_p(h5in, ev2, sig2)
        theta = ots.get_theta(p1, p2)
        axis  = ots.get_axis (p1, p2)

        ampls_1, counts_1, pos_1_cart, pos_1_cyl, qs_1 = ots.charges_pass_thr(h5in,
                                                                              true_file,
                                                                              ev1,
                                                                              ave_true1,
                                                                              rpos_threshold,
                                                                              phi_threshold,
                                                                              zpos_threshold,
                                                                              e_threshold,
                                                                              sens_pos,
                                                                              sens_pos_cyl)

        ampls_2, counts_2, pos_2_cart, pos_2_cyl, qs_2 = ots.charges_pass_thr(h5in,
                                                                              true_file,
                                                                              ev2,
                                                                              ave_true2,
                                                                              rpos_threshold,
                                                                              phi_threshold,
                                                                              zpos_threshold,
                                                                              e_threshold,
                                                                              sens_pos,
                                                                              sens_pos_cyl)

        sigma_phi1 = None
        sigma_phi2 = None

        _, var_phi1 = ats.get_r_and_var_phi(ave_true1, pos_1_cyl[0], qs_1[0])
        sigma_phi1  = np.sqrt(var_phi1)

        _, var_phi2 = ats.get_r_and_var_phi(ave_true2, pos_2_cyl[0], qs_2[0])
        sigma_phi2  = np.sqrt(var_phi2)


        if ampls_1[3]>1000 and ampls_2[3]>1000 and sigma_phi1 and sigma_phi2 and len(qs_1[1]) and len(qs_2[1]) and len(qs_1[2]) and len(qs_2[2]):
            reco1_r            = Rpos(sigma_phi1).value
            reco1_cart_for_phi = ats.barycenter_3D(pos_1_cart[1], qs_1[1])
            reco1_cyl_for_phi  = ots.get_coord_cyl(reco1_cart_for_phi)
            reco1_cart_for_z   = ats.barycenter_3D(pos_1_cart[2], qs_1[2])

            reco_cyl1             = np.array([reco1_r, reco1_cyl_for_phi[1], reco1_cart_for_z[2]])
            reco_cart1            = ots.get_coord_cart(reco_cyl1)
            true1_r, true1_phi, _ = ots.get_coord_cyl(ave_true1)

            reco2_r               = Rpos(sigma_phi2).value
            reco2_cart_for_phi    = ats.barycenter_3D(pos_2_cart[1], qs_2[1])

            reco2_cyl_for_phi     = ots.get_coord_cyl(reco2_cart_for_phi)
            reco2_cart_for_z      = ats.barycenter_3D(pos_2_cart[2], qs_2[2])
            reco_cyl2             = np.array([reco2_r, reco2_cyl_for_phi[1], reco2_cart_for_z[2]])
            reco_cart2            = ots.get_coord_cart(reco_cyl2)

            ## ROTATION:
            reco_cart2_rot        = ots.apply_rot(theta, axis, reco_cart2)
            reco_cyl2_rot         = ots.get_coord_cyl(reco_cart2_rot)
            ave_true2_rot         = ots.apply_rot(theta, axis, ave_true2)
            true2_r, true2_phi, _ = ots.get_coord_cyl(ave_true2_rot)

            print(reco_cart2_rot)
            reco_r1   .append(reco1_r)
            reco_phi1 .append(reco1_cyl_for_phi[1])
            reco_z1   .append(reco1_cart_for_z[2])
            reco_x1   .append(reco_cart1[0])
            reco_y1   .append(reco_cart1[1])

            true_r1   .append(true1_r)
            true_phi1 .append(true1_phi)
            true_z1   .append(ave_true1[2])
            true_x1   .append(ave_true1[0])
            true_y1   .append(ave_true1[1])
            charge1   .append(ampls_1[3])
            reco_r2   .append(reco_cyl2_rot[0])
            reco_phi2 .append(reco_cyl2_rot[1])
            reco_z2   .append(reco_cyl2_rot[2])
            reco_x2   .append(reco_cart2_rot[0])
            reco_y2   .append(reco_cart2_rot[1])

            true_r2   .append(true2_r)
            true_phi2 .append(true2_phi)
            true_z2   .append(ave_true2_rot[2])
            true_x2   .append(ave_true2_rot[0])
            true_y2   .append(ave_true2_rot[1])
            charge2   .append(ampls_2[3])

            event_number1 = h5in.root.MC.extents[ev1]['evt_number']
            event_number2 = h5in.root.MC.extents[ev2]['evt_number']
            events1   .append(event_number1)
            events2   .append(event_number2)
            tot_charge.append(ampls_1[3]+ampls_2[3])

        else:
            reco_r1   .append(1.e9)
            reco_phi1 .append(1.e9)
            reco_z1   .append(1.e9)
            reco_x1   .append(1.e9)
            reco_y1   .append(1.e9)
            true_r1   .append(1.e9)
            true_phi1 .append(1.e9)
            true_z1   .append(1.e9)
            true_x1   .append(1.e9)
            true_y1   .append(1.e9)
            charge1   .append(1.e9)

            reco_r2   .append(1.e9)
            reco_phi2 .append(1.e9)
            reco_z2   .append(1.e9)
            reco_x2   .append(1.e9)
            reco_y2   .append(1.e9)
            true_r2   .append(1.e9)
            true_phi2 .append(1.e9)
            true_z2   .append(1.e9)
            true_x2   .append(1.e9)
            true_y2   .append(1.e9)
            charge2   .append(1.e9)

            events1   .append(1.e9)
            events2   .append(1.e9)
            tot_charge.append(1.e9)


sel1 = (np.array(true_r1) < 1.e9)
a_true_r1    = np.array(true_r1)   [sel1]
a_true_phi1  = np.array(true_phi1) [sel1]
a_true_z1    = np.array(true_z1)   [sel1]
a_true_x1    = np.array(true_x1)   [sel1]
a_true_y1    = np.array(true_y1)   [sel1]
a_true_r2    = np.array(true_r2)   [sel1]
a_true_phi2  = np.array(true_phi2) [sel1]
a_true_z2    = np.array(true_z2)   [sel1]
a_true_x2    = np.array(true_x2)   [sel1]
a_true_y2    = np.array(true_y2)   [sel1]

a_reco_r1    = np.array(reco_r1)   [sel1]
a_reco_phi1  = np.array(reco_phi1) [sel1]
a_reco_z1    = np.array(reco_z1)   [sel1]
a_reco_x1    = np.array(reco_x1)   [sel1]
a_reco_y1    = np.array(reco_y1)   [sel1]
a_reco_r2    = np.array(reco_r2)   [sel1]
a_reco_phi2  = np.array(reco_phi2) [sel1]
a_reco_z2    = np.array(reco_z2)   [sel1]
a_reco_x2    = np.array(reco_x2)   [sel1]
a_reco_y2    = np.array(reco_y2)   [sel1]

a_charge1    = np.array(charge1)   [sel1]
a_charge2    = np.array(charge2)   [sel1]
a_events1    = np.array(events1)   [sel1]
a_events2    = np.array(events2)   [sel1]
a_tot_charge = np.array(tot_charge)[sel1]


np.savez(evt_file,    a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1,
                      a_true_x1=a_true_x1,   a_true_y1=a_true_y1,
                      a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
                      a_true_x2=a_true_x2,   a_true_y2=a_true_y2,
                      a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
                      a_reco_x1=a_reco_x1,   a_reco_y1=a_reco_y1,
                      a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
                      a_reco_x2=a_reco_x2,   a_reco_y2=a_reco_y2,
                      a_charge1=a_charge1,   a_charge2=a_charge2,  a_events1=a_events1,
                      a_events2=a_events2, a_tot_charge=a_tot_charge, tot_evts=tot_evts)



print(datetime.datetime.now())

