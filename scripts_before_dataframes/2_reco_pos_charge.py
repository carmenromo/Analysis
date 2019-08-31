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
from   invisible_cities.io  .mcinfo_io  import read_mcinfo
from   invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

"""
Example of calling this script:
python 2_reco_pos_charge.py 3000 1 6 0 /Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton full_ring_iradius165mm_z140mm_depth3cm_pitch7mm /Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton data_test irad165mm_depth3cm
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
evt_file   = f"{data_path}/full_ring_{identifier}_reco_pos_charge_{start}_{numb}_{nsteps}_{thr_start}"

rpos_threshold = 3
phi_threshold  = 5
zpos_threshold = 4
e_threshold    = 2

rpos_file   = f"{base_path}/r_sigma_phi_table_{identifier}_thr{rpos_threshold}pes_compton_sel_photp.h5"
print(rpos_file)
#Rpos        = ats.load_rpos(rpos_file, group="Radius", node=f"f{rpos_threshold}pes150bins")
Rpos        = ats.load_rpos(rpos_file, group="Radius", node=f"f3pes150bins")

reco_r1, reco_r2, true_r1, true_r2          = [], [], [], []
reco_phi1, reco_phi2, true_phi1, true_phi2  = [], [], [], []
reco_z1, reco_z2, true_z1, true_z2          = [], [], [], []
events, sns_response1, sns_response2        = [], [], []

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

    charge_range   = (1000, 1400)

    for evt in range(events_in_file):
        #for evt in range(1000):
        event_number   = h5in.root.MC.extents[evt]['evt_number']
        this_event_wvf = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')
        sns_over_thr, charges_over_thr = rf.find_SiPMs_over_threshold(this_event_wvf, e_threshold)
        if len(charges_over_thr) == 0: continue

        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        i1, i2, pos_true1, pos_true2, _, _, q1, q2, pos1, pos2 = rf.select_true_pos_from_charge(sns_over_thr, charges_over_thr, charge_range, sens_pos, part_dict)

        if i1 and i2:
            positions1, qs1 = rf.reco_pos_single(pos_true1, np.array(q1), np.array(pos1), rpos_threshold, phi_threshold, zpos_threshold)
            positions2, qs2 = rf.reco_pos_single(pos_true2, np.array(q2), np.array(pos2), rpos_threshold, phi_threshold, zpos_threshold)

            if len(positions1) == 0 or len(positions2) == 0:
                continue

            phi1        = ats.from_cartesian_to_cyl(positions1[0])[:,1]
            var_phi1    = rf.get_var_phi(phi1, qs1[0])
            sigma_phi1  = np.sqrt(var_phi1)
            reco1_r     = Rpos(sigma_phi1).value

            phi2        = ats.from_cartesian_to_cyl(positions2[0])[:,1]
            var_phi2    = rf.get_var_phi(phi2, qs2[0])
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

            reco_r1  .append(reco1_r)
            reco_phi1.append(reco1_phi)
            reco_z1  .append(reco1_z)
            reco_r2  .append(reco2_r)
            reco_phi2.append(reco2_phi)
            reco_z2  .append(reco2_z)
            true_r1  .append(true1_r)
            true_phi1.append(true1_phi)
            true_z1  .append(true1_z)
            true_r2  .append(true2_r)
            true_phi2.append(true2_phi)
            true_z2  .append(true2_z)

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

print(datetime.datetime.now())
