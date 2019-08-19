import os
import sys
import math
import datetime
import sc_utils
import tables         as tb
import numpy          as np
import pandas         as pd
import reco_functions as rf

from   antea.io.mc_io_tb                import read_SiPM_bin_width_from_conf
from   antea.io.mc_io_tb                import go_through_file
from   invisible_cities.io  .mcinfo_io  import read_mcinfo
from   invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

"""
Example of calling this script:
python 1_maps_r.py 3000 1 6 0 /Users/carmenromoluque/nexus_petit_analysis/PETit-ring/Christoff_sim/compton full_ring_iradius165mm_z140mm_depth3cm_pitch7mm data_test
"""

arguments  = sc_utils.parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
nsteps     = arguments.n_steps
thr_start  = arguments.thr_start
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path
evt_file   = f"{data_path}/full_ring_p7mm_d4cm_mapsr_{start}_{numb}_{nsteps}_{thr_start}"

true_r1        = [[] for i in range(0, nsteps)]
true_r2        = [[] for i in range(0, nsteps)]
var_phi1       = [[] for i in range(0, nsteps)]
var_phi2       = [[] for i in range(0, nsteps)]

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    print(f'Trying file {true_file}')

    try:
        h5in = tb.open_file(true_file, mode='r')
        print(f'Analyzing file {true_file}')
    except OSError:
        continue

    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = rf.sensor_position    (h5in)
    sens_pos_cyl   = rf.sensor_position_cyl(h5in)
    bin_width      = read_SiPM_bin_width_from_conf(h5in)

    charge_range   = (1000, 1400)

    for evt in range(events_in_file):
        try:
            ave_true1, ave_true2 = rf.true_photoelect(h5in, true_file, evt, compton=True)
            if len(ave_true1)==0 and len(ave_true2)==0:
                continue

            this_event_wvf = go_through_file(h5in, h5in.root.MC.waveforms, (evt, evt+1), bin_width, 'data')

            sns_dict    = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), sns_dict.values())))
            sns_ids     = np.array(list(sns_dict.keys()))

            for threshold in range(thr_start, nsteps + thr_start):
                indices_over_thr = (tot_charges > threshold)
                sns_over_thr     = sns_ids    [indices_over_thr]
                charges_over_thr = tot_charges[indices_over_thr]

                if len(charges_over_thr) == 0:
                    continue

                ampl1, count1, pos1, pos1_cyl, q1 = rf.sensors_info(ave_true1,
                                                                    sens_pos,
                                                                    sens_pos_cyl,
                                                                    sns_over_thr,
                                                                    charges_over_thr)

                ampl2, count2, pos2, pos2_cyl, q2 = rf.sensors_info(ave_true2,
                                                                    sens_pos,
                                                                    sens_pos_cyl,
                                                                    sns_over_thr,
                                                                    charges_over_thr)

                def lists_var_phi_and_r(var_phi, true_r, phi, r):
                    var_phi[threshold].append(phi)
                    true_r [threshold].append(r)
                    return var_phi, true_r


                def get_true_r(ampl, q, ave_true, pos_cyl, var_phi, true_r):
                    if ampl>charge_range[0] and ampl<charge_range[1] and sum(q) != 0:
                        r, v_phi        = rf.get_r_and_var_phi(ave_true, pos_cyl, q)
                        return lists_var_phi_and_r(var_phi, true_r, v_phi, r)
                    else:
                        return lists_var_phi_and_r(var_phi, true_r, 1.e9, 1.e9)

                var_phi1, true_r1 = get_true_r(ampl1, q1, ave_true1, pos1_cyl, var_phi1, true_r1)
                var_phi2, true_r2 = get_true_r(ampl2, q2, ave_true2, pos2_cyl, var_phi2, true_r2)

        except SipmEmptyList:
            continue


for i in range(nsteps):
    sel1 = (np.array(true_r1[i]) < 1.e9)
    sel2 = (np.array(true_r2[i]) < 1.e9)

    true_r1   [i] = np.array(true_r1 [i])[sel1]
    true_r2   [i] = np.array(true_r2 [i])[sel2]
    var_phi1  [i] = np.array(var_phi1[i])[sel1]
    var_phi2  [i] = np.array(var_phi2[i])[sel2]


np.savez(evt_file, a_true_r1_0=true_r1[0], a_true_r2_0=true_r2[0], a_var_phi1_0=var_phi1[0],
         a_var_phi2_0=var_phi2[0], a_true_r1_1=true_r1[1], a_true_r2_1=true_r2[1], a_var_phi1_1=var_phi1[1],
         a_var_phi2_1=var_phi2[1], a_true_r1_2=true_r1[2], a_true_r2_2=true_r2[2], a_var_phi1_2=var_phi1[2],
         a_var_phi2_2=var_phi2[2], a_true_r1_3=true_r1[3], a_true_r2_3=true_r2[3], a_var_phi1_3=var_phi1[3],
         a_var_phi2_3=var_phi2[3], a_true_r1_4=true_r1[4], a_true_r2_4=true_r2[4], a_var_phi1_4=var_phi1[4],
         a_var_phi2_4=var_phi2[4], a_true_r1_5=true_r1[5], a_true_r2_5=true_r2[5], a_var_phi1_5=var_phi1[5],
         a_var_phi2_5=var_phi2[5])

print(datetime.datetime.now())
