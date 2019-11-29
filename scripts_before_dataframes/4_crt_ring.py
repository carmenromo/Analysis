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

python 4_crt_ring.py 0 1 6 0 /data5/users/carmenromo/PETALO/PETit/PETit-ring/no_refl_sipms/data_ring/ full_ring_iradius165mm_z140mm_depth3cm_pitch7mm_no_refl_sipms /data5/users/carmenromo/PETALO/PETit/PETit-ring/no_refl_sipms/ 4_data_crt_no_compton irad165mm_d3cm_no_refl_sipms
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
evt_file   = f"{data_path}/full_ring_{identifier}_crt_{start}_{numb}_{nsteps}_{thr_start}"

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


time_diff = []
pos_cart1 = []
pos_cart2 = []
event_ids = []

ave_speed_in_LXe = 0.210 # mm/ps 

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

    sens_pos     = rf.sensor_position    (h5in)
    sens_pos_cyl = rf.sensor_position_cyl(h5in)
    bin_width    = read_SiPM_bin_width_from_conf(h5in)

    charge_range   = (0, 10000)
    n_coinc_true = 0
    n_coinc_reco = 0
    for evt in range(events_in_file):
        #for evt in range(1000):

        ave_true1, ave_true2 = rf.true_photoelect(h5in, filename, evt, compton=False)
        if len(ave_true1)==0 or len(ave_true2)==0:
            continue
        n_coinc_true += 1
        event_number       = h5in.root.MC.extents[evt]['evt_number']
        this_event_wvf     = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')
        this_event_wvf_tof = read_mcTOFsns_response(filename, (evt, evt+1))

        sns_over_thr, charges_over_thr     = rf.find_SiPMs_over_threshold(this_event_wvf, e_threshold)
        if len(charges_over_thr) == 0: continue

        sns_closest1 = rf.find_closest_sipm(ave_true1[0], ave_true1[1], ave_true1[2],
                                            sens_pos, sns_over_thr, charges_over_thr)
        sns_closest2 = rf.find_closest_sipm(ave_true2[0], ave_true2[1], ave_true2[2],
                                            sens_pos, sns_over_thr, charges_over_thr)

        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        i1, i2, pos_true1, pos_true2, _, _, ids1, ids2, q1, q2, pos1, pos2 = rf.select_true_pos_from_charge(sns_over_thr, charges_over_thr, charge_range, sens_pos, part_dict)

        sns_dict_tof          = list(this_event_wvf_tof.values())[0]
        tot_charges_tof       = np.array(list(map(lambda x: sum(x.charges), sns_dict_tof.values())))
        first_timestamp_tof   = np.array(list(map(lambda x:     x.times[0], sns_dict_tof.values())))
        sns_ids_tof           = np.array(list(sns_dict_tof.keys()))

        if i1 and i2:
            n_coinc_reco += 1
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

            pos1_cart = []
            pos2_cart = []

            if r1 and phi1 and z1 and len(q1) and r2 and phi2 and z2 and len(q2):
                pos1_cart.append(r1 * np.cos(phi1))
                pos1_cart.append(r1 * np.sin(phi1))
                pos1_cart.append(z1)
                pos2_cart.append(r2 * np.cos(phi2))
                pos2_cart.append(r2 * np.sin(phi2))
                pos2_cart.append(z2)
            a_cart1 = np.array(pos1_cart)
            a_cart2 = np.array(pos2_cart)

            ### Distance between interaction point and sensor detecting first photon
            dist1 = np.linalg.norm(a_cart1 - sens_pos[-min_sns_1])
            dist2 = np.linalg.norm(a_cart2 - sens_pos[-min_sns_2])
            dist  = dist1 - dist2

            delta_t = 1/2 *(dist/ave_speed_in_LXe - min_time)

            time_diff.append(delta_t)
            pos_cart1.append(a_cart1)
            pos_cart2.append(a_cart2)
            event_ids.append(event_number)

a_time_diff = np.array(time_diff)
a_pos_cart1 = np.array(pos_cart1)
a_pos_cart2 = np.array(pos_cart2)
a_event_ids = np.array(event_ids)

np.savez(evt_file, time_diff=a_time_diff, pos_cart1=a_pos_cart1, pos_cart2=a_pos_cart2, event_ids=a_event_ids)

print(datetime.datetime.now())

