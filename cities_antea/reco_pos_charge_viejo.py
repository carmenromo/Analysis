import os
import sys
import math
import datetime
import tables         as tb
import numpy          as np
import pandas         as pd

from antea.reco                       import reco_functions      as rf
from antea.io.mc_io                   import read_mcsns_response
from invisible_cities.io  .mcinfo_io  import read_mcinfo
from invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

start     = int(sys.argv[1])
numb      = int(sys.argv[2])
nsteps    = int(sys.argv[3])
thr_start = int(sys.argv[4])

events_path = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton'
file_name   = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm'
data_path   = '/test'
evt_file    = '{data_path}/full_ring_p7mm_d3cm_mapsr_{start}_{numb}_{steps}_{thr_start}'.format(data_path, start, numb, nsteps, thr_start)

events    = [[] for i in range(0, nsteps)]
true_r1   = [[] for i in range(0, nsteps)]
true_r2   = [[] for i in range(0, nsteps)]
true_phi1 = [[] for i in range(0, nsteps)]
true_phi2 = [[] for i in range(0, nsteps)]
true_z1   = [[] for i in range(0, nsteps)]
true_z2   = [[] for i in range(0, nsteps)]

reco_r1   = [[] for i in range(0, nsteps)]
reco_r2   = [[] for i in range(0, nsteps)]
reco_phi1 = [[] for i in range(0, nsteps)]
reco_phi2 = [[] for i in range(0, nsteps)]
reco_z1   = [[] for i in range(0, nsteps)]
reco_z2   = [[] for i in range(0, nsteps)]
charge1   = [[] for i in range(0, nsteps)]
charge2   = [[] for i in range(0, nsteps)]

for number in range(start, start+numb):
    number_str = f'{:03d}'.format(number)
    true_file  = f'{events_path}/{file_name}.{number_str}.pet.h5'
    print(f'Analyzing file {true_file}')

    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        continue

    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = rf.sensor_position    (h5in)
    sens_pos_cyl   = rf.sensor_position_cyl(h5in)

    for evt in range(events_in_file):
        try:
            ave_true1, ave_true2 = rf.true_photoelect(h5in, true_file, evt, compton=False)

            if not len(ave_true1) and not len(ave_true2):
                continue

            this_event_wvf = read_mcsns_response(true_file, (evt, evt+1))
            event_number   = h5in.root.MC.extents[evt]['evt_number']

            sns_dict    = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), sns_dict.values())))
            sns_ids     = np.array(list(sns_dict.keys()))

            for threshold in range(thr_start, nsteps + thr_start):
                indices_over_thr = (tot_charges > threshold)
                sns_over_thr     = sns_ids    [indices_over_thr]
                charges_over_thr = tot_charges[indices_over_thr]

                if len(charges_over_thr) == 0:
                    continue

                ### THRESHOLD FOR R:
                ampl1, ampl2, count1, count2, _, _, pos1_r, pos2_r, q1, q2 = ats.sensor_classification(i1, i2,
                                                                                                       ave_true1,
                                                                                                       ave_true2,
                                                                                                       sens_pos,
                                                                                                       sens_pos_cyl,
                                                                                                       sns_over_thr,
                                                                                                       charges_over_thr)

                sigma_phi1 = sigma_phi2 = None

                if ampl1 > 1000 and sum(q1) != 0:
                    _, var_phi = ats.get_r_and_var_phi(ave_true1, pos1_r, q1)
                    sigma_phi1 = np.sqrt(var_phi)

                if ampl2 > 1000 and sum(q2) != 0:
                    _, var_phi = ats.get_r_and_var_phi(ave_true2, pos2_r, q2)
                    sigma_phi2 = np.sqrt(var_phi)


                ### THRESHOLD FOR PHI:
                indices_over_thr_phi = (tot_charges >= threshold)
                sns_over_thr_phi     = sns_ids    [indices_over_thr_phi]
                charges_over_thr_phi = tot_charges[indices_over_thr_phi]
                        
                if len(charges_over_thr_phi) == 0:
                    continue
                        
                _, _, _, _, pos1_for_phi, pos2_for_phi, _, _, q1_for_phi, q2_for_phi = ats.sensor_classification(i1, i2,
                                                                                                                 ave_true1,
                                                                                                                 ave_true2,
                                                                                                                 sens_pos,
                                                                                                                 sens_pos_cyl,
                                                                                                                 sns_over_thr_phi,
                                                                                                                 charges_over_thr_phi)

                ### THRESHOLD FOR Z:
                indices_over_thr_z = (tot_charges >= threshold)
                sns_over_thr_z     = sns_ids    [indices_over_thr_z]
                charges_over_thr_z = tot_charges[indices_over_thr_z]

                if len(charges_over_thr_z) == 0:
                    continue

                _, _, _, _, pos1_for_z, pos2_for_z, _, _, q1_for_z, q2_for_z = ats.sensor_classification(i1, i2,
                                                                                                         ave_true1,
                                                                                                         ave_true2,
                                                                                                         sens_pos,
                                                                                                         sens_pos_cyl,
                                                                                                         sns_over_thr_z,
                                                                                                         charges_over_thr_z)

                ### THRESHOLD FOR E:
                indices_over_thr_e = (tot_charges >= threshold)
                sns_over_thr_e     = sns_ids    [indices_over_thr_e]
                charges_over_thr_e = tot_charges[indices_over_thr_e]

                if len(charges_over_thr_e) == 0:
                    continue
                
                ampl1_e, ampl2_e, _, _, _, _, _, _, _, _ = ats.sensor_classification(i1, i2,
                                                                                     ave_true1,
                                                                                     ave_true2,
                                                                                     sens_pos,
                                                                                     sens_pos_cyl,
                                                                                     sns_over_thr_e,
                                                                                     charges_over_thr_e)

                if len(q1_for_phi) and len(q1_for_z) and sigma_phi1 and ampl1_e:
                    reco_r            = Rpos(sigma_phi1).value
                    reco_cart_for_phi = ats.barycenter_3D(pos1_for_phi, q1_for_phi)
                    reco_cyl_for_phi  = ats.get_coord_cyl(reco_cart_for_phi)
                    reco_cart_for_z   = ats.barycenter_3D(pos1_for_z  , q1_for_z  )

                    true_r   = np.sqrt((ave_true1[0]*ave_true1[0])+(ave_true1[1]*ave_true1[1]))
                    true_phi = np.arctan2(ave_true1[1], ave_true1[0])

                    reco_r1  [threshold].append(reco_r)
                    reco_phi1[threshold].append(reco_cyl_for_phi[1])
                    reco_z1  [threshold].append(reco_cart_for_z [2])
                    true_r1  [threshold].append(true_r)
                    true_phi1[threshold].append(true_phi)
                    true_z1  [threshold].append(ave_true1[2])
                    charge1  [threshold].append(ampl1_e)
                    events   [threshold].append(event_number)
                else:
                    reco_r1  [threshold].append(1.e9)
                    reco_phi1[threshold].append(1.e9)
                    reco_z1  [threshold].append(1.e9)
                    true_r1  [threshold].append(1.e9)
                    true_phi1[threshold].append(1.e9)
                    true_z1  [threshold].append(1.e9)
                    charge1  [threshold].append(1.e9)
                    events   [threshold].append(event_number)

                if len(q2_for_phi) and len(q2_for_z) and sigma_phi2 and ampl2_e:
                    reco_r            = Rpos(sigma_phi2).value
                    reco_cart_for_phi = ats.barycenter_3D(pos2_for_phi, q2_for_phi)
                    reco_cyl_for_phi  = ats.get_coord_cyl(reco_cart_for_phi)
                    reco_cart_for_z   = ats.barycenter_3D(pos2_for_z  , q2_for_z  )

                    true_r   = np.sqrt((ave_true2[0]*ave_true2[0])+(ave_true2[1]*ave_true2[1]))
                    true_phi = np.arctan2(ave_true2[1], ave_true2[0])

                    reco_r2  [threshold].append(reco_r)
                    reco_phi2[threshold].append(reco_cyl_for_phi[1])
                    reco_z2  [threshold].append(reco_cart_for_z[2])
                    true_r2  [threshold].append(true_r)
                    true_phi2[threshold].append(true_phi)
                    true_z2  [threshold].append(ave_true2[2])
                    charge2  [threshold].append(ampl2_e)
                else:
                    reco_r2  [threshold].append(1.e9)
                    reco_phi2[threshold].append(1.e9)
                    reco_z2  [threshold].append(1.e9)
                    true_r2  [threshold].append(1.e9)
                    true_phi2[threshold].append(1.e9)
                    true_z2  [threshold].append(1.e9)
                    charge2  [threshold].append(1.e9)

        except SipmEmptyList:
            continue


for i in range(nsteps):
    sel1 = (np.array(true_r1[i]) < 1.e9)
    sel2 = (np.array(true_r2[i]) < 1.e9)

    events    [i] = np.array(events    [i])[sel1]
    true_r1   [i] = np.array(true_r1   [i])[sel1]
    true_r2   [i] = np.array(true_r2   [i])[sel2]
    true_phi1 [i] = np.array(true_phi1 [i])[sel1]
    true_phi2 [i] = np.array(true_phi2 [i])[sel2]
    true_z1   [i] = np.array(true_z1   [i])[sel1]
    true_z2   [i] = np.array(true_z2   [i])[sel2]

    reco_r1   [i] = np.array(reco_r1   [i])[sel1]
    reco_r2   [i] = np.array(reco_r2   [i])[sel2]
    reco_phi1 [i] = np.array(reco_phi1 [i])[sel1]
    reco_phi2 [i] = np.array(reco_phi2 [i])[sel2]
    reco_z1   [i] = np.array(reco_z1   [i])[sel1]
    reco_z2   [i] = np.array(reco_z2   [i])[sel2]
    charge1   [i] = np.array(charge1   [i])[sel1]
    charge2   [i] = np.array(charge2   [i])[sel2]

print(datetime.datetime.now())

np.savez(evt_file, a_events_0=events[0], a_true_r1_0=true_r1[0], a_true_r2_0=true_r2[0], a_true_phi1_0=true_phi1[0],
         a_true_phi2_0=true_phi2[0], a_true_z1_0=true_z1[0], a_true_z2_0=true_z2[0], a_reco_r1_0=reco_r1[0],
         a_reco_r2_0=reco_r2[0], a_reco_phi1_0=reco_phi1[0], a_reco_phi2_0=reco_phi2[0], a_reco_z1_0=reco_z1[0],
         a_reco_z2_0=reco_z2[0], a_charge1_0=charge1[0], a_charge2_0=charge2[0], a_events_1=events[1], a_true_r1_1=true_r1[1],
         a_true_r2_1=true_r2[1], a_true_phi1_1=true_phi1[1], a_true_phi2_1=true_phi2[1], a_true_z1_1=true_z1[1],
         a_true_z2_1=true_z2[1], a_reco_r1_1=reco_r1[1], a_reco_r2_1=reco_r2[1], a_reco_phi1_1=reco_phi1[1],
         a_reco_phi2_1=reco_phi2[1], a_reco_z1_1=reco_z1[1], a_reco_z2_1=reco_z2[1], a_charge1_1=charge1[1],
         a_charge2_1=charge2[1], a_events_2=events[2], a_true_r1_2=true_r1[2], a_true_r2_2=true_r2[2],
         a_true_phi1_2=true_phi1[2], a_true_phi2_2=true_phi2[2], a_true_z1_2=true_z1[2], a_true_z2_2=true_z2[2],
         a_reco_r1_2=reco_r1[2], a_reco_r2_2=reco_r2[2], a_reco_phi1_2=reco_phi1[2], a_reco_phi2_2=reco_phi2[2],
         a_reco_z1_2=reco_z1[2], a_reco_z2_2=reco_z2[2], a_charge1_2=charge1[2], a_charge2_2=charge2[2],
         a_events_3=events[3], a_true_r1_3=true_r1[3], a_true_r2_3=true_r2[3], a_true_phi1_3=true_phi1[3],
         a_true_phi2_3=true_phi2[3], a_true_z1_3=true_z1[3], a_true_z2_3=true_z2[3], a_reco_r1_3=reco_r1[3],
         a_reco_r2_3=reco_r2[3], a_reco_phi1_3=reco_phi1[3], a_reco_phi2_3=reco_phi2[3], a_reco_z1_3=reco_z1[3],
         a_reco_z2_3=reco_z2[3], a_charge1_3=charge1[3], a_charge2_3=charge2[3], a_events_4=events[4],
         a_true_r1_4=true_r1[4], a_true_r2_4=true_r2[4], a_true_phi1_4=true_phi1[4], a_true_phi2_4=true_phi2[4],
         a_true_z1_4=true_z1[4], a_true_z2_4=true_z2[4], a_reco_r1_4=reco_r1[4], a_reco_r2_4=reco_r2[4],
         a_reco_phi1_4=reco_phi1[4], a_reco_phi2_4=reco_phi2[4], a_reco_z1_4=reco_z1[4], a_reco_z2_4=reco_z2[4],
         a_charge1_4=charge1[4], a_charge2_4=charge2[4], a_events_5=events[5], a_true_r1_5=true_r1[5],
         a_true_r2_5=true_r2[5], a_true_phi1_5=true_phi1[5], a_true_phi2_5=true_phi2[5], a_true_z1_5=true_z1[5],
         a_true_z2_5=true_z2[5], a_reco_r1_5=reco_r1[5], a_reco_r2_5=reco_r2[5], a_reco_phi1_5=reco_phi1[5],
         a_reco_phi2_5=reco_phi2[5], a_reco_z1_5=reco_z1[5], a_reco_z2_5=reco_z2[5], a_charge1_5=charge1[5],
         a_charge2_5=charge2[5])
