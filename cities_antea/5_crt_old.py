import sys
import math
import datetime
import tables         as tb
import numpy          as np
import analysis_utils as ats

from antea.reco     import reco_functions        as rf
from antea.io.mc_io import read_mcsns_response
from antea.io.mc_io import read_mcTOFsns_response

from invisible_cities.io  .mcinfo_io  import read_mcinfo
from invisible_cities.core.exceptions import SipmEmptyList
from invisible_cities.core.exceptions import SipmZeroCharge
from invisible_cities.core.exceptions import SensorBinningNotFound

print(datetime.datetime.now())

start     = int(sys.argv[1])
numb      = int(sys.argv[2])

events_path = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton'
file_name   = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm'
data_path   = '/test'
evt_file    = f'{data_path}/full_ring_p7mm_d3cm_mapsr_{start}_{numb}_{nsteps}_{thr_start}'
rpos_file   = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/refl_walls/r_sigma_phi_table_iradius165mm_thr3pes_ref_walls_compton_sel_photopeak.h5'
Rpos        = ats.load_rpos(rpos_file, group = "Radius", node  = "f3pes150bins")

rpos_threshold = 3
phi_threshold  = 5
zpos_threshold = 4
e_threshold    = 3

time_diff = []
pos_cart1 = []
pos_cart2 = []
events    = []

speed_in_vacuum  = 0.299792458 # mm/ps
ave_speed_in_LXe = 0.210 # mm/ps

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

    for evt in range(events_in_file):
        try:
            this_event_dict    = read_mcinfo(h5in, (evt, evt+1))
            this_event_wvf     = read_mcsns_response(true_file, (evt, evt+1))
            this_event_wvf_tof = read_mcTOFsns_response(true_file, (evt, evt+1))
            part_dict          = list(this_event_dict.values())[0]

            i1, i2, ave_true1, ave_true2 = ats.true_photoelect_compton(h5in, true_file, evt)

            ## ONLY COINCIDENCES ARE TAKEN
            if i1 and i2:
                sns_dict_tof          = list(this_event_wvf_tof.values())[0]
                tot_charges_tof       = np.array(list(map(lambda x: sum(x.charges), sns_dict_tof.values())))
                first_timestamp_tof   = np.array(list(map(lambda x:     x.times[0], sns_dict_tof.values())))
                sns_ids_tof           = np.array(list(sns_dict_tof.keys()))

                ## First we store all charges > 0
                sns_dict          = list(this_event_wvf.values())[0]
                tot_charges       = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
                sns_ids           = np.array(list(sns_dict.keys()))

                ### THRESHOLD FOR R
                indices_over_thr = (tot_charges > rpos_threshold)
                sns_over_thr     = sns_ids    [indices_over_thr]
                charges_over_thr = tot_charges[indices_over_thr]

                sns_closest1 = ats.find_closest_sipm(ave_true1[0], ave_true1[1], ave_true1[2],
                                                     sens_pos, sns_over_thr, charges_over_thr)
                sns_closest2 = ats.find_closest_sipm(ave_true2[0], ave_true2[1], ave_true2[2],
                                                     sens_pos, sns_over_thr, charges_over_thr)

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

                pos1_cart = []
                pos2_cart = []

                if ampl1_e>1000 and ampl2_e>1000 and sigma_phi1 and sigma_phi2 and len(q1_for_phi) and len(q2_for_phi) and len(q1_for_z) and len(q2_for_z):
                    reco1_r            = Rpos(sigma_phi1).value
                    reco1_cart_for_phi = ats.barycenter_3D(pos1_for_phi, q1_for_phi)
                    reco1_cyl_for_phi  = ats.get_coord_cyl(reco1_cart_for_phi)
                    reco1_cart_for_z   = ats.barycenter_3D(pos1_for_z  , q1_for_z  )

                    pos1_cart.append(reco1_r * np.cos(reco1_cyl_for_phi[1]))
                    pos1_cart.append(reco1_r * np.sin(reco1_cyl_for_phi[1]))
                    pos1_cart.append(reco1_cart_for_z[2])


                    reco2_r            = Rpos(sigma_phi2).value
                    reco2_cart_for_phi = ats.barycenter_3D(pos2_for_phi, q2_for_phi)
                    reco2_cyl_for_phi  = ats.get_coord_cyl(reco2_cart_for_phi)
                    reco2_cart_for_z   = ats.barycenter_3D(pos2_for_z  , q2_for_z  )

                    pos2_cart.append(reco2_r * np.cos(reco2_cyl_for_phi[1]))
                    pos2_cart.append(reco2_r * np.sin(reco2_cyl_for_phi[1]))
                    pos2_cart.append(reco2_cart_for_z[2])

                a_cart1 = np.array(pos1_cart)
                a_cart2 = np.array(pos2_cart)


                tof = {}
                for sns_id, timestamp in zip(sns_ids_tof, first_timestamp_tof):
                    if timestamp > 0.:
                        tof[sns_id] = timestamp

                tof_1 = {}
                tof_2 = {}

                for sns_id, t in tof.items():
                    pos     = sens_pos    [-sns_id]
                    pos_cyl = sens_pos_cyl[-sns_id]

                    pos_closest = sens_pos[sns_closest1]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.:
                        tof_1[sns_id] = t

                    pos_closest = sens_pos[sns_closest2]
                    scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
                    if scalar_prod > 0.:
                        tof_2[sns_id] = t


                min_sns_1  = min(tof_1, key=tof_1.get)
                min_time_1 = tof_1[min_sns_1]
                min_sns_2  = min(tof_2, key=tof_2.get)
                min_time_2 = tof_2[min_sns_2]
                min_time   = min_time_1 - min_time_2


                ### Distance between interaction point and sensor detecting first photon
                dist1 = np.linalg.norm(a_cart1 - sens_pos[-min_sns_1])
                dist2 = np.linalg.norm(a_cart2 - sens_pos[-min_sns_2])
                dist  = dist1 - dist2

                ### Distance of the interaction point from the centre of the system
                inter1 = np.linalg.norm(a_cart1)
                inter2 = np.linalg.norm(a_cart2)
                inter  = inter1 - inter2

                delta_t = 1/2 *(min_time - inter/speed_in_vacuum - dist/ave_speed_in_LXe)
                print(delta_t)
                print(evt)
                print('')
                time_diff.append(delta_t)
                pos_cart1.append(a_cart1)
                pos_cart2.append(a_cart2)
                events   .append(evt)

        except ValueError:
            continue
        except OSError:
            continue
        except SipmEmptyList:
            continue
        except SensorBinningNotFound:
            continue


a_time_diff = np.array(time_diff)
a_pos_cart1 = np.array(pos_cart1)
a_pos_cart2 = np.array(pos_cart2)
a_events    = np.array(events)

np.savez(evt_file, a_time_diff=a_time_diff, a_pos_cart1=a_pos_cart1, a_pos_cart2=a_pos_cart2, a_events=a_events)

print(datetime.datetime.now())


