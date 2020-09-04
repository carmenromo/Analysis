import os
import sys
import math
import tables as tb
import numpy  as np
import pandas as pd
import datetime
import analysis_utils as ats

from   antea.io.mc_io                       import read_mcsns_response
from   invisible_cities.io.mcinfo_io        import read_mcinfo
import invisible_cities.reco.dst_functions  as     dstf

from invisible_cities.core.exceptions import SipmEmptyList
from invisible_cities.core.exceptions import SipmZeroCharge

print(datetime.datetime.now())

start     = int(sys.argv[1])
numb      = int(sys.argv[2])
nsteps    = int(sys.argv[3])
thr_start = int(sys.argv[4])

eventsPath = '/data5/users/carmenromo/tof_setup/data_tof'
file_name  = 'pet_box_setup'
evt_file   = '/data5/users/carmenromo/tof_setup/pet_box_setup_xmap_{0}_{1}_{2}_{3}'.format(start, numb, nsteps, thr_start)

true_x1        = [[] for i in range(0, nsteps)]
true_x2        = [[] for i in range(0, nsteps)]
var_r1         = [[] for i in range(0, nsteps)]
var_r2         = [[] for i in range(0, nsteps)]
touched_sipms1 = [[] for i in range(0, nsteps)]
touched_sipms2 = [[] for i in range(0, nsteps)]


for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = '%s/%s_%s.pet.h5'%(eventsPath, file_name, number_str)
    print('Analyzing file {0}'.format(true_file))

    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        print('File {0} does not exist'.format(true_file))
        continue

    h5extents      = h5in.root.MC.extents
    events_in_file = len(h5extents)
    sens_pos       = ats.sensor_position(h5in)

    for evt in range(events_in_file):
        try:
            this_event_dict = read_mcinfo(h5in, (evt, evt+1))
            this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
            event_number    = h5in.root.MC.extents[evt]['evt_number']
            part_dict       = list(this_event_dict.values())[0]

            i1, i2, ave_true1, ave_true2 = ats.true_photoelect_setup(h5in, true_file, evt)

            if not i1 and not i2:
                continue
            if i1 and i2 and (not len(ave_true1) or not len(ave_true2)):
                continue

            sns_dict    = list(this_event_wvf.values())[0]
            tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
            sns_ids     = np.array(list(sns_dict.keys()))

            for threshold in range(thr_start, nsteps+thr_start):
                indices_over_thr = (tot_charges > threshold)
                sns_over_thr     = sns_ids    [indices_over_thr]
                charges_over_thr = tot_charges[indices_over_thr]

                if len(charges_over_thr) == 0:
                    continue

                ampl1, ampl2, count1, count2, pos1, pos2, pos1_r, pos2_r,  q1, q2 = ats.sensor_classification_cart(i1, i2,
                                                                                                                   ave_true1,
                                                                                                                   ave_true2,
                                                                                                                   sens_pos,
                                                                                                                   sns_over_thr,
                                                                                                                   charges_over_thr)

                if ampl1 != 0 and sum(q1) != 0:
                    mean_r = np.average(pos1_r, weights=q1)
                    var_r  = np.average((pos1_r - mean_r)**2, weights=q1)
                    var_r1        [threshold].append(var_r)
                    true_x1       [threshold].append(ave_true1[0])
                    touched_sipms1[threshold].append(count1)
                else:
                    var_r1        [threshold].append(1.e9)
                    true_x1       [threshold].append(1.e9)
                    touched_sipms1[threshold].append(1.e9)
            
                if ampl2 != 0 and sum(q2) != 0:
                    mean_r = np.average(pos2_r, weights=q2)
                    var_r  = np.average((pos2_r - mean_r)**2, weights=q2)
                    var_r2        [threshold].append(var_r)
                    true_x2       [threshold].append(ave_true2[0])
                    touched_sipms2[threshold].append(count2)
                else:
                    var_r2        [threshold].append(1.e9)
                    true_x2       [threshold].append(1.e9)
                    touched_sipms2[threshold].append(1.e9)
                
        except ValueError:
            continue
        except OSError:
            continue
        except SipmEmptyList:
            continue


for i in range(nsteps):
    sel1 = (np.array(true_x1[i]) < 1.e9)
    sel2 = (np.array(true_x2[i]) < 1.e9)

    true_x1       [i] = np.array(true_x1       [i])[sel1]
    true_x2       [i] = np.array(true_x2       [i])[sel2]
    var_r1        [i] = np.array(var_r1        [i])[sel1]
    var_r2        [i] = np.array(var_r2        [i])[sel2]
    touched_sipms1[i] = np.array(touched_sipms1[i])[sel1]
    touched_sipms2[i] = np.array(touched_sipms2[i])[sel2]

np.savez(evt_file, a_true_x1_0=true_x1[0], a_true_x2_0=true_x2[0], a_var_r1_0=var_r1[0], a_var_r2_0=var_r2[0], 
         a_touched_sipms1_0=touched_sipms1[0], a_touched_sipms2_0=touched_sipms2[0], a_true_x1_1=true_x1[1], 
         a_true_x2_1=true_x2[1], a_var_r1_1=var_r1[1], a_var_r2_1=var_r2[1], a_touched_sipms1_1=touched_sipms1[1], 
         a_touched_sipms2_1=touched_sipms2[1], a_true_x1_2=true_x1[2], a_true_x2_2=true_x2[2], a_var_r1_2=var_r1[2], 
         a_var_r2_2=var_r2[2], a_touched_sipms1_2=touched_sipms1[2], a_touched_sipms2_2=touched_sipms2[2], 
         a_true_x1_3=true_x1[3], a_true_x2_3=true_x2[3], a_var_r1_3=var_r1[3], a_var_r2_3=var_r2[3], 
         a_touched_sipms1_3=touched_sipms1[3], a_touched_sipms2_3=touched_sipms2[3], a_true_x1_4=true_x1[4], 
         a_true_x2_4=true_x2[4], a_var_r1_4=var_r1[4], a_var_r2_4=var_r2[4], a_touched_sipms1_4=touched_sipms1[4],
         a_touched_sipms2_4=touched_sipms2[4], a_true_x1_5=true_x1[5], a_true_x2_5=true_x2[5], a_var_r1_5=var_r1[5], 
         a_var_r2_5=var_r2[5], a_touched_sipms1_5=touched_sipms1[5], a_touched_sipms2_5=touched_sipms2[5])

print(datetime.datetime.now())
print(true_x1, var_r1)
