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

correction_filename  = base_path + '/corrections_full_ring_irad165cm_d3cm_pitch7mm_photoel.h5'
corrections0         = ats.load_zr_corrections(correction_filename,
                                               group = "ZRcorrections",
                                               node = f"GeometryE_2.0mm_2.0mm",
                                               norm_strategy = "index",
                                               norm_opts = {"index": (10,10)})  #Choose the bin for normalization (zbins,rbins)
corrections1         = ats.load_zr_corrections(correction_filename,
                                               group = "ZRcorrections",
                                               node = f"GeometryE_2.5mm_2.5mm",
                                               norm_strategy = "index",
                                               norm_opts = {"index": (7,9)})
corrections2         = ats.load_zr_corrections(correction_filename,
                                               group = "ZRcorrections",
                                               node = f"GeometryE_5.0mm_5.0mm",
                                               norm_strategy = "index",
                                               norm_opts = {"index": (7,4)})

rpos_threshold = 3
phi_threshold  = 5
zpos_threshold = 4
e_threshold    = 3

non_corrected_charge1 = []
non_corrected_charge2 = []
corrected_charge1_0   = []
corrected_charge1_1   = []
corrected_charge1_2   = []
corrected_charge2_0   = []
corrected_charge2_1   = []
corrected_charge2_2   = []

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
                reco_cart_for_z   = ats.barycenter_3D(pos1_for_z, q1_for_z)
                corr_charge1_0    = ampl1_e * corrections0(reco_cart_for_z[2], reco_r).value
                corr_charge1_1    = ampl1_e * corrections1(reco_cart_for_z[2], reco_r).value
                corr_charge1_2    = ampl1_e * corrections2(reco_cart_for_z[2], reco_r).value

                non_corrected_charge1.append(ampl1_e)
                corrected_charge1_0  .append(corr_charge1_0)
                corrected_charge1_1  .append(corr_charge1_1)
                corrected_charge1_2  .append(corr_charge1_2)
            else:
                non_corrected_charge1.append(1.e9)
                corrected_charge1_0  .append(1.e9)
                corrected_charge1_1  .append(1.e9)
                corrected_charge1_2  .append(1.e9)


            if ampl2_e>1000 and len(q2_for_phi) and len(q2_for_z) and sigma_phi2:
                reco_r            = Rpos(sigma_phi2).value
                reco_cart_for_z   = ats.barycenter_3D(pos2_for_z, q2_for_z)
                corr_charge2_0    = ampl2_e * corrections0(reco_cart_for_z[2], reco_r).value
                corr_charge2_1    = ampl2_e * corrections1(reco_cart_for_z[2], reco_r).value
                corr_charge2_2    = ampl2_e * corrections2(reco_cart_for_z[2], reco_r).value

                non_corrected_charge2.append(ampl2_e)
                corrected_charge2_0  .append(corr_charge2_0)
                corrected_charge2_1  .append(corr_charge2_1)
                corrected_charge2_2  .append(corr_charge2_2)

            else:
                non_corrected_charge2.append(1.e9)
                corrected_charge2_0  .append(1.e9)
                corrected_charge2_1  .append(1.e9)
                corrected_charge2_2  .append(1.e9)


        except ValueError:
            continue
        except OSError:
            continue
        except SipmEmptyList:
            continue

sel1 = (np.array(non_corrected_charge1) < 1.e9)
sel2 = (np.array(non_corrected_charge2) < 1.e9)

a_non_corrected_charge1 = np.array(non_corrected_charge1)[sel1]
a_corrected_charge1_0   = np.array(corrected_charge1_0)  [sel1]
a_corrected_charge1_1   = np.array(corrected_charge1_1)  [sel1]
a_corrected_charge1_2   = np.array(corrected_charge1_2)  [sel1]

a_non_corrected_charge2 = np.array(non_corrected_charge2)[sel2]
a_corrected_charge2_0   = np.array(corrected_charge2_0)  [sel2]
a_corrected_charge2_1   = np.array(corrected_charge2_1)  [sel2]
a_corrected_charge2_2   = np.array(corrected_charge2_2)  [sel2]

np.savez(evt_file, a_non_corrected_charge1=a_non_corrected_charge1, a_non_corrected_charge2=a_non_corrected_charge2,
         a_corrected_charge1_0=a_corrected_charge1_0, a_corrected_charge2_0=a_corrected_charge2_0,
         a_corrected_charge1_1=a_corrected_charge1_1, a_corrected_charge2_1=a_corrected_charge2_1,
         a_corrected_charge1_2=a_corrected_charge1_2, a_corrected_charge2_2=a_corrected_charge2_2)


print(datetime.datetime.now())



