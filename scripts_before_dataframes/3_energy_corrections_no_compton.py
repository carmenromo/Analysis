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
python 3_energy_corrections_no_compton.py 0 1 6 0 /data5/users/carmenromo/PETALO/PETit/PETit-ring/no_refl_sipms/data_ring/ full_ring_iradius165mm_z140mm_depth3cm_pitch7mm_no_refl_sipms /data5/users/carmenromo/PETALO/PETit/PETit-ring/no_refl_sipms/ 3_data_correction_energy_no_compton irad165mm_d3cm_no_refl_sipms
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
evt_file   = f"{data_path}/full_ring_{identifier}_energy_corr_{start}_{numb}_{nsteps}_{thr_start}"

correction_filename  = base_path + f"/corrections_full_ring_{identifier}_no_compton.h5"

if identifier == 'irad165mm_depth3cm':
    bin_pos = [(20,10), (20, 7), (11, 4)]
elif identifier == 'irad165mm_d3cm_refl_walls':
    bin_pos = [(20,10), (20, 7), (11, 4)]
elif identifier == 'irad165mm_d3cm_no_refl_sipms':
    bin_pos = [(20, 6), (20, 5), (11, 4)]
elif identifier == 'irad165mm_d4cm':
    bin_pos = [(20,12), (20,10), (11, 5)]

corrections0 = ats.load_zr_corrections(correction_filename,
                                       group = "ZRcorrections",
                                       node = f"GeometryE_2.0mm_2.0mm",
                                       norm_strategy = "index",
                                       norm_opts = {"index": bin_pos[0]})
corrections1 = ats.load_zr_corrections(correction_filename,
                                       group = "ZRcorrections",
                                       node = f"GeometryE_2.5mm_2.5mm",
                                       norm_strategy = "index",
                                       norm_opts = {"index": bin_pos[1]})
corrections2 = ats.load_zr_corrections(correction_filename,
                                       group = "ZRcorrections",
                                       node = f"GeometryE_5.0mm_5.0mm",
                                       norm_strategy = "index",
                                       norm_opts = {"index": bin_pos[2]})

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


non_corrected_charge1, non_corrected_charge2 = [], []
corrected_charge1_0  , corrected_charge2_0   = [], []
corrected_charge1_1  , corrected_charge2_1   = [], []
corrected_charge1_2  , corrected_charge2_2   = [], []

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

    for evt in range(events_in_file):
        #for evt in range(1000):

        ave_true1, ave_true2 = rf.true_photoelect(h5in, filename, evt, compton=False)
        if len(ave_true1)==0 and len(ave_true2)==0:
            continue

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

            if np.isnan(reco1_r) or np.isnan(reco2_r):
                continue
            reco_cart = ats.barycenter_3D(positions1[1], qs1[1])
            reco1_phi = np.arctan2(reco_cart[1], reco_cart[0])

            reco_cart = ats.barycenter_3D(positions2[1], qs2[1])
            reco2_phi = np.arctan2(reco_cart[1], reco_cart[0])

            reco_cart = ats.barycenter_3D(positions1[2], qs1[2])
            reco1_z   = reco_cart[2]

            reco_cart = ats.barycenter_3D(positions2[2], qs2[2])
            reco2_z   = reco_cart[2]

            ampl1          = sum(q1)
            corr_charge1_0 = ampl1 * corrections0(reco1_z, reco1_r).value
            corr_charge1_1 = ampl1 * corrections1(reco1_z, reco1_r).value
            corr_charge1_2 = ampl1 * corrections2(reco1_z, reco1_r).value

            ampl2 = sum(q2)
            corr_charge2_0 = ampl2 * corrections0(reco2_z, reco2_r).value
            corr_charge2_1 = ampl2 * corrections1(reco2_z, reco2_r).value
            corr_charge2_2 = ampl2 * corrections2(reco2_z, reco2_r).value

            non_corrected_charge1.append(ampl1)
            corrected_charge1_0.append(corr_charge1_0)
            corrected_charge1_1.append(corr_charge1_1)
            corrected_charge1_2.append(corr_charge1_2)
            corrected_charge2_0.append(corr_charge2_0)
            corrected_charge2_1.append(corr_charge2_1)
            corrected_charge2_2.append(corr_charge2_2)

            #sns_response1.append(q1)
            #sns_response2.append(q2)


a_non_corrected_charge1 = np.array(non_corrected_charge1)
a_non_corrected_charge2 = np.array(non_corrected_charge2)
a_corrected_charge1_0   = np.array(corrected_charge1_0  )
a_corrected_charge2_0   = np.array(corrected_charge2_0  )
a_corrected_charge1_1   = np.array(corrected_charge1_1  )
a_corrected_charge2_1   = np.array(corrected_charge2_1  )
a_corrected_charge1_2   = np.array(corrected_charge1_2  )
a_corrected_charge2_2   = np.array(corrected_charge2_2  )


np.savez(evt_file, a_non_corrected_charge1=non_corrected_charge1, a_non_corrected_charge2=non_corrected_charge2, 
         a_corrected_charge1_0=corrected_charge1_0, a_corrected_charge2_0=corrected_charge2_0, a_corrected_charge1_1=corrected_charge1_1, 
         a_corrected_charge2_1=corrected_charge2_1, a_corrected_charge1_2=corrected_charge1_2, a_corrected_charge2_2=corrected_charge2_2)  


print(datetime.datetime.now())
