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

from   invisible_cities.io  .mcinfo_io  import read_mcinfo
from   invisible_cities.core.exceptions import SipmEmptyList

print(datetime.datetime.now())

"""
Example of calling this script:
python true_information.py 0 1 6 0 /data5/users/carmenromo/PETALO/PETit/PETit-ring/reflect_walls/data_ring/ full_ring_iradius165mm_z140mm_depth3cm_pitch7mm_refl_walls /data5/users/carmenromo/PETALO/PETit/PETit-ring/reflect_walls/ data_true_information irad165mm_d3cm_refl_walls
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
evt_file   = f"{data_path}/full_ring_{identifier}_true_info_{start}_{numb}"


tot_evts     = []
phot_single  = []
phot_coinc   = []
compt_single = []
comp_coinc   = []
energy_phot  = []
energy_compt = []

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

    n_tot_evts     = 0
    n_phot_single  = 0
    n_phot_coinc   = 0
    n_compt_single = 0
    n_comp_coinc   = 0

    for evt in range(events_in_file):
        n_tot_evts += 1

        this_event_dict = read_mcinfo(h5in, (evt, evt+1))
        part_dict       = list(this_event_dict.values())[0]
        
        ave_true11, ave_true12, energy11, energy12 = rf.true_photoelect(h5in, filename, evt, compton=False, energy=True)

        energy_phot.append(energy11)
        energy_phot.append(energy12)

        if len(ave_true11) and len(ave_true12):
            n_phot_coinc += 1
        elif len(ave_true11) or len(ave_true12):
            n_phot_single += 1
        else:
            ave_true21, ave_true22, energy21, energy22 = rf.true_photoelect(h5in, filename, evt, compton=True, energy=True)
            energy_compt.append(energy21)
            energy_compt.append(energy22)

            if len(ave_true21) and len(ave_true22):
                n_comp_coinc += 1
            elif len(ave_true21) or len(ave_true22):
                n_compt_single += 1
            else:
                continue

    tot_evts    .append(n_tot_evts    )
    phot_single .append(n_phot_single )
    phot_coinc  .append(n_phot_coinc  )
    compt_single.append(n_compt_single)
    comp_coinc  .append(n_comp_coinc  )


a_tot_evts     = np.array(tot_evts    )
a_phot_single  = np.array(phot_single )
a_phot_coinc   = np.array(phot_coinc  )
a_compt_single = np.array(compt_single)
a_comp_coinc   = np.array(comp_coinc  )
a_energy_phot  = np.array(energy_phot )
a_energy_compt = np.array(energy_compt)


np.savez(evt_file, a_tot_evts=a_tot_evts, a_phot_single=a_phot_single, a_phot_coinc=a_phot_coinc, a_compt_single=a_compt_single, 
         a_comp_coinc=a_comp_coinc, a_energy_phot=a_energy_phot, a_energy_compt=a_energy_compt)

print(datetime.datetime.now())
