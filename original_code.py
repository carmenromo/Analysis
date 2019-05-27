import os
import sys
import math
import datetime
import tables         as tb
import numpy          as np
import pandas         as pd
import reco_functions as rf

from   antea.io.mc_io                   import read_mcsns_response
from   invisible_cities.io  .mcinfo_io  import read_mcinfo
from   invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

start     = int(sys.argv[1])
numb      = int(sys.argv[2])
nsteps    = int(sys.argv[3])
thr_start = int(sys.argv[4])

events_path = '/data4/PETALO/PETit-ring/7mm_pitch'
file_name   = 'full_ring_iradius165mm_z140mm_depth3cm_pitch7mm'
data_path   = '/data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton/1_data_maps_r'
evt_file    = '{0}/full_ring_p7mm_d3cm_mapsr_{1}_{2}_{3}_{4}'.format(data_path, start, numb, nsteps, thr_start)


def true_photoelect(h5in, true_file, evt, compton=False):
    """Returns the position of the true photoelectric energy deposition
       calculated with barycenter algorithm.
       It allows the possibility of including compton events.
       """

    this_event_dict = read_mcinfo        (     h5in, (evt, evt+1))
    this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
    part_dict       = list(this_event_dict.values())[0]

    ave_true1 = []
    ave_true2 = []

    for indx, part in part_dict.items():
        if part.name == 'e-' :
            mother = part_dict[part.mother_indx]
            if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                if mother.primary and np.isclose(mother.E*1000., 510.999, atol=1.e-3):
                    if compton==True: pass
                    else:
                        if np.isclose(sum(h.E for h in part.hits), 0.476443, atol=1.e-6): pass
                        else: continue

                    if mother.p[1] > 0.: ave_true1 = get_true_pos(part)
                    else:                ave_true2 = get_true_pos(part)
    return ave_true1, ave_true2

true_r1        = [[] for i in range(0, nsteps)]
true_r2        = [[] for i in range(0, nsteps)]
var_phi1       = [[] for i in range(0, nsteps)]
var_phi2       = [[] for i in range(0, nsteps)]

for number in range(start, start+numb):
    number_str = f''"{:03d}".format(number)
    true_file  = f'{events_path}/{file_name}.{number_str}.pet.h5'
    print(f'Analyzing file {true_file}')

    h5in           = tb.open_file(true_file, mode='r')
    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)

    sens_pos       = rf.sensor_position    (h5in)
    sens_pos_cyl   = rf.sensor_position_cyl(h5in)

    for evt in range(events_in_file):
        try:
            ave_true1, ave_true2 = true_photoelect(h5in, true_file, evt, compton=False)

            if not len(ave_true1) and not len(ave_true2):
                continue

            this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))

            sns_dict    = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
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


                if ampl1 and sum(q1) != 0:
                    r1, var_phi = rf.get_r_and_var_phi(ave_true1, pos1_phi, q1)
                    var_phi1[threshold].append(var_phi)
                    true_r1 [threshold].append(r1)
                else:
                    var_phi1[threshold].append(1.e9)
                    true_r1 [threshold].append(1.e9)

                if ampl2 and sum(q2) != 0:
                    r2, var_phi = rf.get_r_and_var_phi(ave_true2, pos2_phi, q2)
                    var_phi2[threshold].append(var_phi)
                    true_r2 [threshold].append(r2)
                else:
                    var_phi2[threshold].append(1.e9)
                    true_r2 [threshold].append(r2)

        except ValueError:
            continue
        except OSError:
            continue
        except SipmEmptyList:
            continue


for i in range(nsteps):
    sel1 = (np.array(true_r1[i]) < 1.e9)
    sel2 = (np.array(true_r2[i]) < 1.e9)

    true_r1   [i] = np.array(true_r1 [i])[sel1]
    true_r2   [i] = np.array(true_r2 [i])[sel2]
    var_phi1  [i] = np.array(var_phi1[i])[sel1]
    var_phi2  [i] = np.array(var_phi2[i])[sel2]
