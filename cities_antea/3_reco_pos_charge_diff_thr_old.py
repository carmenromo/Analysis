import os
import sys
import math
import datetime
import tables         as tb
import numpy          as np
import pandas         as pd
import analysis_utils as ats

from antea.reco                       import reco_functions      as rf
from antea.io.mc_io                   import read_mcsns_response
from invisible_cities.io  .mcinfo_io  import read_mcinfo
from invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

start     = int(sys.argv[1])
numb      = int(sys.argv[2])

events_path = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton'
file_name   = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm'
data_path   = '/test'
evt_file    = f'{data_path}/full_ring_p7mm_d3cm_mapsr_{start}_{numb}_{nsteps}_{thr_start}'
rpos_file   = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/refl_walls/r_sigma_phi_table_iradius165mm_thr3pes_ref_walls_compton_sel_photopeak.h5'
Rpos        = ats.load_rpos(rpos_file, group = "Radius", node  = "f3pes150bins")

true_r1    = []
true_r2    = []
true_phi1  = []
true_phi2  = []
true_z1    = []
true_z2    = []

reco_r1    = []
reco_r2    = []
reco_phi1  = []
reco_phi2  = []
reco_z1    = []
reco_z2    = []

charge1    = []
charge2    = []
events1    = []
events2    = []
tot_charge = []
tot_evts   = 0

rpos_threshold = 3
zpos_threshold = 4
phi_threshold  = 5
e_threshold    = 3

for number in range(start, start+numb):
    number_str = '{:03d}'.format(number)
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

    #for evt in range(events_in_file):
    for evt in range(1000):
        try:
            this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
            event_number    = h5in.root.MC.extents[evt]['evt_number']

            i1, i2, ave_true1, ave_true2 = ats.true_photoelect_compton(h5in, true_file, evt)

            if not i1 and not i2:
                continue

            sns_dict    = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), sns_dict.values())))
            sns_ids     = np.array(list(sns_dict.keys()))

            ### THRESHOLD FOR R
            indices_over_thr = (tot_charges > rpos_threshold)
            sns_over_thr     = sns_ids    [indices_over_thr]
            charges_over_thr = tot_charges[indices_over_thr]

            if len(charges_over_thr) == 0:
                continue

            ampl1, ampl2, count1, count2, _, _, pos1_r, pos2_r, q1, q2 = ats.sensor_classification(i1, i2,
                                                                                                   ave_true1,
                                                                                                   ave_true2,
                                                                                                   sens_pos,
                                                                                                   sens_pos_cyl,
                                                                                                   sns_over_thr,
                                                                                                   charges_over_thr)

            sigma_phi1 = sigma_phi2 = None

            if ampl1 != 0 and sum(q1) != 0:
                _, var_phi = ats.get_r_and_var_phi(ave_true1, pos1_r, q1)
                sigma_phi1 = np.sqrt(var_phi)

            if ampl2 != 0 and sum(q2) != 0:
                _, var_phi = ats.get_r_and_var_phi(ave_true2, pos2_r, q2)
                sigma_phi2 = np.sqrt(var_phi)


            ### THRESHOLD FOR PHI
            indices_over_thr_phi = (tot_charges >= phi_threshold)
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

            ### THRESHOLD FOR Z
            indices_over_thr_z = (tot_charges >= zpos_threshold)
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

            ### THRESHOLD FOR E
            indices_over_thr_e = (tot_charges >= e_threshold)
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

            if ampl1_e>1000 and len(q1_for_phi) and len(q1_for_z) and sigma_phi1:
                reco_r            = Rpos(sigma_phi1).value
                reco_cart_for_phi = ats.barycenter_3D(pos1_for_phi, q1_for_phi)
                reco_cyl_for_phi  = ats.get_coord_cyl(reco_cart_for_phi)
                reco_cart_for_z   = ats.barycenter_3D(pos1_for_z  , q1_for_z  )

                true_r   = np.sqrt((ave_true1[0]*ave_true1[0])+(ave_true1[1]*ave_true1[1]))
                true_phi = np.arctan2(ave_true1[1], ave_true1[0])

                reco_r1   .append(reco_r)
                reco_phi1 .append(reco_cyl_for_phi[1])
                reco_z1   .append(reco_cart_for_z [2])
                true_r1   .append(true_r)
                true_phi1 .append(true_phi)
                true_z1   .append(ave_true1[2])
                charge1   .append(ampl1_e)
                events1   .append(event_number)

            else:
                reco_r1   .append(1.e9)
                reco_phi1 .append(1.e9)
                reco_z1   .append(1.e9)
                true_r1   .append(1.e9)
                true_phi1 .append(1.e9)
                true_z1   .append(1.e9)
                charge1   .append(1.e9)
                events1   .append(1.e9)

            if ampl2_e>1000 and len(q2_for_phi) and len(q2_for_z) and sigma_phi2:
                reco_r            = Rpos(sigma_phi2).value
                reco_cart_for_phi = ats.barycenter_3D(pos2_for_phi, q2_for_phi)
                reco_cyl_for_phi  = ats.get_coord_cyl(reco_cart_for_phi)
                reco_cart_for_z   = ats.barycenter_3D(pos2_for_z  , q2_for_z  )

                true_r   = np.sqrt((ave_true2[0]*ave_true2[0])+(ave_true2[1]*ave_true2[1]))
                true_phi = np.arctan2(ave_true2[1], ave_true2[0])

                reco_r2   .append(reco_r)
                reco_phi2 .append(reco_cyl_for_phi[1])
                reco_z2   .append(reco_cart_for_z [2])
                true_r2   .append(true_r)
                true_phi2 .append(true_phi)
                true_z2   .append(ave_true2[2])
                charge2   .append(ampl2_e)
                events2   .append(event_number)

            else:
                reco_r2   .append(1.e9)
                reco_phi2 .append(1.e9)
                reco_z2   .append(1.e9)
                true_r2   .append(1.e9)
                true_phi2 .append(1.e9)
                true_z2   .append(1.e9)
                charge2   .append(1.e9)
                events2   .append(1.e9)

            if ampl1_e>1000 and ampl2_e>1000:
                tot_charge.append(ampl1_e+ampl2_e)


        except ValueError:
            continue
        except OSError:
            continue
        except SipmEmptyList:
            continue

sel1 = (np.array(true_r1) < 1.e9)
sel2 = (np.array(true_r2) < 1.e9)
a_true_r1    = np.array(true_r1)  [sel1]
a_true_phi1  = np.array(true_phi1)[sel1]
a_true_z1    = np.array(true_z1)  [sel1]
a_true_r2    = np.array(true_r2)  [sel2]
a_true_phi2  = np.array(true_phi2)[sel2]
a_true_z2    = np.array(true_z2)  [sel2]

a_reco_r1    = np.array(reco_r1)  [sel1]
a_reco_phi1  = np.array(reco_phi1)[sel1]
a_reco_z1    = np.array(reco_z1)  [sel1]
a_reco_r2    = np.array(reco_r2)  [sel2]
a_reco_phi2  = np.array(reco_phi2)[sel2]
a_reco_z2    = np.array(reco_z2)  [sel2]

a_charge1    = np.array(charge1)  [sel1]
a_charge2    = np.array(charge2)  [sel2]
a_events1    = np.array(events1)  [sel1]
a_events2    = np.array(events2)  [sel2]
a_tot_charge = np.array(tot_charge)


np.savez(evt_file,    a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1,
         a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2,
         a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1,
         a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,
         a_charge1=a_charge1,   a_charge2=a_charge2,  a_events1=a_events1,
         a_events2=a_events2, a_tot_charge=a_tot_charge, tot_evts=tot_evts)

print(datetime.datetime.now())




