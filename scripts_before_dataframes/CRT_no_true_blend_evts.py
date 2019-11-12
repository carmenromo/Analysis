import os
import sys
import math
import datetime
import sc_utils
import tables         as tb
import numpy          as np
import pandas         as pd
import reco_functions as rf
import analysis_utils as ats

from   antea.io.mc_io_tb                import read_SiPM_bin_width_from_conf
from   antea.io.mc_io_tb                import go_through_file
from   antea.io.mc_io_tb                import read_mcTOFsns_response
from   invisible_cities.io  .mcinfo_io  import read_mcinfo
from   invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

"""
Example of calling this script:

python CRT.py 0 1 6 0 /data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton/analysis/data_ring full_ring_iradius165mm_z140mm_depth3cm_pitch7mm /data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton/analysis/ 4_data_crt_no_compton irad165mm_depth3cm
"""

arguments  = sc_utils.parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
nsteps     = arguments.n_steps
thr_start  = arguments.thr_start
eventsPath = arguments.events_path
file_name  = arguments.file_name
base_path  = arguments.base_path
data_path  = arguments.data_path
identifier = arguments.identifier

data_path  = f"{base_path}/{data_path}"
evt_file   = f"{data_path}/full_ring_{identifier}_crt_{start}_{numb}"

if identifier == 'irad165mm_d3cm_no_refl_sipms':
    rpos_threshold = 4
else:
    rpos_threshold = 3

phi_threshold  = 5
zpos_threshold = 4
e_threshold    = 2

rpos_file   = f"{base_path}/r_sigma_phi_table_{identifier}_thr{rpos_threshold}pes_no_compton.h5"
#Rpos        = ats.load_rpos(rpos_file, group="Radius", node=f"f{rpos_threshold}pes150bins")

if identifier == 'irad165mm_d3cm_no_refl_sipms':
    Rpos = ats.load_rpos(rpos_file, group="Radius", node=f"f4pes150bins")
else:
    Rpos = ats.load_rpos(rpos_file, group="Radius", node=f"f3pes150bins")


time_diff1 = []
time_diff2 = []
time_diff3 = []
pos_cart1  = []
pos_cart2  = []
event_ids  = []

ave_speed_in_LXe = 0.210 # mm/ps 
speed_in_vacuum  = 0.299792458 # mm/ps

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    try:
        print('Trying file {0}'.format(filename))
        h5in = tb.open_file(filename, mode='r')
    except ValueError:
        continue
    except OSError:
         continue
    print('Analyzing file {0}'.format(filename))

    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = rf.sensor_position    (h5in)
    sens_pos_cyl   = rf.sensor_position_cyl(h5in)
    bin_width      = read_SiPM_bin_width_from_conf(h5in)

    charge_range = (1000, 1400)

    for evt in range(events_in_file):
        #for evt in range(1000):

        event_number       = h5in.root.MC.extents[evt]['evt_number']
        this_event_wvf     = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')
        this_event_wvf_tof = read_mcTOFsns_response(filename, (evt, evt+1))
        sns_over_thr, charges_over_thr = rf.find_SiPMs_over_threshold(this_event_wvf, e_threshold)

        if len(charges_over_thr) == 0: continue

        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        i1, i2, pos_true1, pos_true2, _, _, ids1, ids2, q1, q2, pos1, pos2 = rf.select_true_pos_from_charge(sns_over_thr, 
                                                                                                            charges_over_thr, 
                                                                                                            charge_range, 
                                                                                                            sens_pos, 
                                                                                                            part_dict)

        if i1 and i2:
            positions1, qs1 = rf.reco_pos_single(pos_true1, np.array(q1), np.array(pos1), rpos_threshold, phi_threshold, zpos_threshold)
            positions2, qs2 = rf.reco_pos_single(pos_true2, np.array(q2), np.array(pos2), rpos_threshold, phi_threshold, zpos_threshold)

            if len(positions1) == 0 or len(positions2) == 0:
                continue

            phi1s       = ats.from_cartesian_to_cyl(positions1[0])[:,1]
            var_phi1    = rf.get_var_phi(phi1s, qs1[0])
            sigma_phi1  = np.sqrt(var_phi1)
            r1          = Rpos(sigma_phi1).value

            phi2s       = ats.from_cartesian_to_cyl(positions2[0])[:,1]
            var_phi2    = rf.get_var_phi(phi2s, qs2[0])
            sigma_phi2  = np.sqrt(var_phi2)
            r2          = Rpos(sigma_phi2).value

            reco_cart = ats.barycenter_3D(positions1[1], qs1[1])
            phi1      = np.arctan2(reco_cart[1], reco_cart[0])

            reco_cart = ats.barycenter_3D(positions2[1], qs2[1])
            phi2      = np.arctan2(reco_cart[1], reco_cart[0])

            reco_cart = ats.barycenter_3D(positions1[2], qs1[2])
            z1        = reco_cart[2]

            reco_cart = ats.barycenter_3D(positions2[2], qs2[2])
            z2        = reco_cart[2]

            pos1_cart = []
            pos2_cart = []
            if r1 and phi1 and z1 and len(q1) and r2 and phi2 and z2 and len(q2):
                pos1_cart.append(r1 * np.cos(phi1))
                pos1_cart.append(r1 * np.sin(phi1))
                pos1_cart.append(z1)
                pos2_cart.append(r2 * np.cos(phi2))
                pos2_cart.append(r2 * np.sin(phi2))
                pos2_cart.append(z2)
            else: continue

            sns_dict_tof = list(this_event_wvf_tof.values())[0]

            ### CAREFUL, I AM BLENDING THE EVENTS!!!
            if evt%2 == 0:
                a_cart1 = np.array(pos1_cart)
                a_cart2 = np.array(pos2_cart)
                min_t1, min_id1 = rf.find_first_time_of_sensors(sns_dict_tof, ids1)
                min_t2, min_id2 = rf.find_first_time_of_sensors(sns_dict_tof, ids2)
            else:
                a_cart1 = np.array(pos2_cart)
                a_cart2 = np.array(pos1_cart)
                min_t1, min_id1 = rf.find_first_time_of_sensors(sns_dict_tof, ids2)
                min_t2, min_id2 = rf.find_first_time_of_sensors(sns_dict_tof, ids1)


            min_t1 = min_t1/0.001 #from ns to ps
            min_t2 = min_t2/0.001

            ### Distance between interaction point and sensor detecting first photon
            dp1 = np.linalg.norm(a_cart1 - sens_pos[-min_id1])
            dp2 = np.linalg.norm(a_cart2 - sens_pos[-min_id2])

            ### Distance between interaction point and center of the geometry
            geo_center = np.array([0,0,0])
            dg1 = np.linalg.norm(a_cart1 - geo_center)
            dg2 = np.linalg.norm(a_cart2 - geo_center)
            
            delta_t1 = 1/2 *(min_t2 - min_t1 + (dp1 - dp2)/ave_speed_in_LXe)
            delta_t2 = 1/2 *(min_t2 - min_t1)
            delta_t3 = 1/2 *(min_t2 - min_t1 + (dp1 - dp2)/ave_speed_in_LXe + (dg1 - dg2)/speed_in_vacuum)

            time_diff1.append(delta_t1)
            time_diff2.append(delta_t2)
            time_diff3.append(delta_t3)
            pos_cart1 .append(a_cart1)
            pos_cart2 .append(a_cart2)
            event_ids .append(event_number)

a_time_diff1 = np.array(time_diff1)
a_time_diff2 = np.array(time_diff2)
a_time_diff3 = np.array(time_diff3)
a_pos_cart1  = np.array(pos_cart1 )
a_pos_cart2  = np.array(pos_cart2 )
a_event_ids  = np.array(event_ids )

np.savez(evt_file, time_diff1=a_time_diff1, time_diff2=a_time_diff2, time_diff3=a_time_diff3, pos_cart1=a_pos_cart1, pos_cart2=a_pos_cart2, event_ids=a_event_ids)

print(datetime.datetime.now())
