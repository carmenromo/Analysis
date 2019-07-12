import os
import sys
import math
import datetime
import tables             as tb
import numpy              as np
import analysis_utils     as ats
import optimization_utils as ots

from   antea.io   .mc_io                import read_mcsns_response
from   invisible_cities.io  .mcinfo_io  import read_mcinfo

from invisible_cities.core.exceptions   import SipmEmptyList
from invisible_cities.core.exceptions   import SipmZeroCharge

print(datetime.datetime.now())


start     = int(sys.argv[1])
numb      = int(sys.argv[2])

eventsPath = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton/'
file_name  = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm'
base_path  = '/Users/carmenromoluque/Analysis/Christoph_code'
data_path  = base_path + '/data_test'
evt_file   = '{0}/full_ring_irad15cm_d3cm_p7mm_reco_info_{1}_{2}'.format(data_path, start, numb)
rpos_file  = eventsPath+'/r_sigma_phi_table_iradius165mm_thr4pes_depth3cm_compton_sel_photopeak_new.h5'
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

th_r   = 3
th_phi = 5
th_z   = 4
th_e   = 2


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

    #for evt in range(events_in_file):
    for evt in range(100):
        tot_evts += 1
        try:
            i1, i2, ave_true1, ave_true2 = ats.true_photoelect_compton(h5in, true_file, evt)

            if i1 or i2:
                this_event_dict = read_mcinfo(h5in, (evt, evt+1))
                this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
                event_number    = h5in.root.MC.extents[evt]['evt_number']
                part_dict       = list(this_event_dict.values())[0]

                sns_dict    = list(this_event_wvf.values())[0]
                tot_charges = np.array(list(map(lambda x: sum(x.charges), sns_dict.values())))
                sns_ids     = np.array(list(sns_dict.keys()))

                list_thrs = [th_r, th_phi, th_z, th_e]

                ampls1    = []
                ampls2    = []
                counts1   = []
                counts2   = []
                poss1     = []
                poss2     = []
                poss_cyl1 = []
                poss_cyl2 = []
                qs1       = []
                qs2       = []

                for i, th in enumerate(list_thrs):
                    sigma_phi1 = None
                    sigma_phi2 = None

                    list_charges = ats.create_list_all_charges(this_event_wvf[event_number], th)

                    indices_over_thr = (tot_charges > th)
                    sns_over_thr     = sns_ids    [indices_over_thr]
                    charges_over_thr = tot_charges[indices_over_thr]

                    if len(charges_over_thr) == 0:
                        continue

                    pos1, pos2, pos_cyl1, pos_cyl2, q1, q2, ampl1, ampl2, count1, count2 = ats.sensor_classif(list_charges,
                                                                                                              sns_over_thr,
                                                                                                              charges_over_thr,
                                                                                                              sens_pos,
                                                                                                              sens_pos_cyl,
                                                                                                              n_sipm=5)
                    ampls1   .append(ampl1)
                    counts1  .append(count1)
                    poss1    .append(np.array(pos1))
                    poss_cyl1.append(np.array(pos_cyl1))
                    qs1      .append(np.array(q1))

                    ampls2   .append(ampl2)
                    counts2  .append(count2)
                    poss2    .append(np.array(pos2))
                    poss_cyl2.append(np.array(pos_cyl2))
                    qs2      .append(np.array(q2))

                if len(ave_true1) and ampls1[3] and len(qs1[0]):
                    _, var_phi1 = ats.get_r_and_var_phi(ave_true1, poss_cyl1[0], qs1[0])
                    sigma_phi1  = np.sqrt(var_phi1)

                if len(ave_true2) and ampls2[3] and len(qs2[0]):
                    _, var_phi2 = ats.get_r_and_var_phi(ave_true2, poss_cyl2[0], qs2[0])
                    sigma_phi2  = np.sqrt(var_phi2)



                if sigma_phi1 and ampls1[3]>1000 and len(qs1[1]) and len(qs1[2]):
                    reco1_r            = Rpos(sigma_phi1).value
                    reco1_cart_for_phi = ats.barycenter_3D(poss1[1], qs1[1])
                    reco1_cyl_for_phi  = ots.get_coord_cyl(reco1_cart_for_phi)
                    reco1_cart_for_z   = ats.barycenter_3D(poss1[2], qs1[2])

                    reco_cyl1  = np.array([reco1_r, reco1_cyl_for_phi[1], reco1_cart_for_z[2]])
                    reco_cart1 = ots.get_coord_cart(reco_cyl1)
                    true1_r, true1_phi, _ = ots.get_coord_cyl(ave_true1)

                    reco_r1  .append(reco1_r)
                    reco_phi1.append(reco1_cyl_for_phi[1])
                    reco_z1  .append(reco1_cart_for_z [2])
                    reco_x1  .append(reco_cart1[0])
                    reco_y1  .append(reco_cart1[1])
                    true_r1  .append(true1_r)
                    true_phi1.append(true1_phi)
                    true_z1  .append(ave_true1[2])
                    true_x1  .append(ave_true1[0])
                    true_y1  .append(ave_true1[1])
                    events1  .append(event_number)
                    charge1  .append(ampls1[3])

                else:
                    reco_r1  .append(1.e9)
                    reco_phi1.append(1.e9)
                    reco_z1  .append(1.e9)
                    reco_x1  .append(1.e9)
                    reco_y1  .append(1.e9)
                    true_r1  .append(1.e9)
                    true_phi1.append(1.e9)
                    true_z1  .append(1.e9)
                    true_x1  .append(1.e9)
                    true_y1  .append(1.e9)
                    events1  .append(1.e9)
                    charge1  .append(1.e9)

                if sigma_phi2 and ampls2[3]>1000 and len(qs2[1]) and len(qs2[2]):
                    reco2_r            = Rpos(sigma_phi2).value
                    reco2_cart_for_phi = ats.barycenter_3D(poss2[1], qs2[1])
                    reco2_cyl_for_phi  = ots.get_coord_cyl(reco2_cart_for_phi)
                    reco2_cart_for_z   = ats.barycenter_3D(poss2[2], qs2[2])

                    reco_cyl2  = np.array([reco2_r, reco2_cyl_for_phi[1], reco2_cart_for_z[2]])
                    reco_cart2 = ots.get_coord_cart(reco_cyl2)
                    true2_r, true2_phi, _ = ots.get_coord_cyl(ave_true2)

                    reco_r2  .append(reco2_r)
                    reco_phi2.append(reco2_cyl_for_phi[1])
                    reco_z2  .append(reco2_cart_for_z [2])
                    reco_x2  .append(reco_cart2[0])
                    reco_y2  .append(reco_cart2[1])
                    true_r2  .append(true2_r)
                    true_phi2.append(true2_phi)
                    true_z2  .append(ave_true2[2])
                    true_x2  .append(ave_true2[0])
                    true_y2  .append(ave_true2[1])
                    charge2  .append(ampls2[3])
                    events2  .append(event_number)

                else:
                    reco_r2  .append(1.e9)
                    reco_phi2.append(1.e9)
                    reco_z2  .append(1.e9)
                    reco_x2  .append(1.e9)
                    reco_y2  .append(1.e9)
                    true_r2  .append(1.e9)
                    true_phi2.append(1.e9)
                    true_z2  .append(1.e9)
                    true_x2  .append(1.e9)
                    true_y2  .append(1.e9)
                    charge2  .append(1.e9)
                    events2  .append(1.e9)

                if ampls1[3] and ampls2[3] and sigma_phi1 and sigma_phi2:
                    tot_charge.append(ampls1[3]+ampls2[3])
                else:
                    tot_charge.append(1.e9)


        except ValueError:
            continue
        except SipmEmptyList:
            continue
        except SipmZeroCharge:
            continue

sel1 = (np.array(true_r1) < 1.e9)
sel2 = (np.array(true_r2) < 1.e9)
sel3 = (np.array(tot_charge) < 1.e9)

a_true_r1    = np.array(true_r1)   [sel1]
a_true_phi1  = np.array(true_phi1) [sel1]
a_true_z1    = np.array(true_z1)   [sel1]
a_true_x1    = np.array(true_x1)   [sel1]
a_true_y1    = np.array(true_y1)   [sel1]
a_true_r2    = np.array(true_r2)   [sel2]
a_true_phi2  = np.array(true_phi2) [sel2]
a_true_z2    = np.array(true_z2)   [sel2]
a_true_x2    = np.array(true_x2)   [sel2]
a_true_y2    = np.array(true_y2)   [sel2]

a_reco_r1    = np.array(reco_r1)   [sel1]
a_reco_phi1  = np.array(reco_phi1) [sel1]
a_reco_z1    = np.array(reco_z1)   [sel1]
a_reco_x1    = np.array(reco_x1)   [sel1]
a_reco_y1    = np.array(reco_y1)   [sel1]
a_reco_r2    = np.array(reco_r2)   [sel2]
a_reco_phi2  = np.array(reco_phi2) [sel2]
a_reco_z2    = np.array(reco_z2)   [sel2]
a_reco_x2    = np.array(reco_x2)   [sel2]
a_reco_y2    = np.array(reco_y2)   [sel2]

a_charge1    = np.array(charge1)   [sel1]
a_charge2    = np.array(charge2)   [sel2]
a_events1    = np.array(events1)   [sel1]
a_events2    = np.array(events2)   [sel2]
a_tot_charge = np.array(tot_charge)[sel3]

print(len(a_charge1), len(a_charge2))

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

print(a_tot_charge)
print(tot_evts)
print(datetime.datetime.now())
