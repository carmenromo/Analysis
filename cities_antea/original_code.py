import os
import sys
import math
import datetime
import tables         as tb
import numpy          as np
import pandas         as pd
#import reco_functions as rf

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
evt_file    = '{0}/full_ring_p7mm_d3cm_mapsr_{1}_{2}_{3}_{4}'.format(data_path, start, numb, nsteps, thr_start)

true_r1        = [[] for i in range(0, nsteps)]
true_r2        = [[] for i in range(0, nsteps)]
var_phi1       = [[] for i in range(0, nsteps)]
var_phi2       = [[] for i in range(0, nsteps)]

for number in range(start, start+numb):
    number_str = f''"{:03d}".format(number)
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

            this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))

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
                    if ampl and sum(q) != 0:
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

print(true_r1)
