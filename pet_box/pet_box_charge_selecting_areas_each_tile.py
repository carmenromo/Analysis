import sys
import argparse
import datetime
import tables   as tb
import numpy    as np

import pet_box_functions as pbf

from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcparticles
from antea.io.mc_io import load_mcsns_response
from antea.io.mc_io import load_sns_positions

import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf

""" To run this script
python pet_box_charge_selecting_areas_each_tile.py 0 1 0 5 /Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_h5 PetBox_asymmetric_HamamatsuVUV
/Users/carmenromoluque/nexus_petit_analysis/tof_setup/PetBox_analysis/data_charge
"""

print(datetime.datetime.now())

arguments     = pbf.parse_args(sys.argv)
start         = arguments.first_file
numb          = arguments.n_files
thr_ch_start  = arguments.thr_ch_start
thr_ch_nsteps = arguments.thr_ch_nsteps
in_path       = arguments.in_path
file_name     = arguments.file_name
out_path      = arguments.out_path

## NEW ONES
# area00 = [22, 23, 32, 33]
# area01 = [26, 27, 36, 37]
# area02 = [62, 63, 72, 73]
# area03 = [66, 67, 76, 77]
# area10 = [122, 123, 132, 133]

area00 = [10, 11, 18, 19]
area01 = [14, 15, 22, 23]
area02 = [42, 43, 50, 51]
area03 = [46, 47, 54, 55]
area10 = [75, 74, 83, 82]


areas1 = [area00, area01, area02, area03]

threshold = 2

evt_file   = f'{out_path}/pet_box_charge_select_areas_each_tile_{start}_{numb}_thr{threshold}pes'

chargs_phot   = [[] for i in range(len(areas1)+1)]
truepos_phot  = [[] for i in range(len(areas1)+1)]
id_max        = [[] for i in range(len(areas1)+1)]
touched_sipms = [[] for i in range(len(areas1)+1)]
evt_ids       = []
num_tile      = []

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    true_file  = f'{in_path}/{file_name}.{number_str}.pet.h5'
    try:
        h5in = tb.open_file(true_file, mode='r')
    except OSError:
        print(f'File {true_file} does not exist')
        continue
    print(f'Analyzing file {true_file}')

    mcparticles   = load_mcparticles   (true_file)
    mchits        = load_mchits        (true_file)
    sns_response  = load_mcsns_response(true_file)
    sns_positions = load_sns_positions (true_file)
    DataSiPM      = sns_positions.rename(columns={"sensor_id": "SensorID","x": "X", "y": "Y", "z": "Z"})
    DataSiPM_idx  = DataSiPM.set_index('SensorID')

    events = mcparticles.event_id.unique()
    for evt in events:
        evt_parts = mcparticles [mcparticles .event_id == evt]
        evt_hits  = mchits      [mchits      .event_id == evt]
        evt_sns   = sns_response[sns_response.event_id == evt]

        he_gamma, true_pos_he = pbf.select_gamma_high_energy(evt_parts, evt_hits)
        phot, true_pos_phot   = mcf.select_photoelectric(evt_parts, evt_hits)

        sel_phot0 = np.array([pos[2] for pos in true_pos_phot])
        sel_neg_phot = sel_phot0[sel_phot0<0]
        sel_pos_phot = sel_phot0[sel_phot0>0]
        phot_neg_pos = np.array(true_pos_phot)[sel_phot0<0]
        phot_pos_pos = np.array(true_pos_phot)[sel_phot0>0]

        sel_neg_he = np.array([pos[2] for pos in true_pos_he])
        sel_neg_he = sel_neg_he[sel_neg_he<0]
        sel_pos_he = sel_neg_he[sel_neg_he>0]

        if phot:
            evt_sns = rf.find_SiPMs_over_threshold(evt_sns, threshold=threshold)
            if len(evt_sns) == 0:
                continue
            ids, pos, qs = pbf.info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns)
            if len(qs) == 0:
                continue
            max_charge_s_id = ids[np.argmax(qs)]
            if len(sel_neg_phot)>0: ### Be careful with the meaning of this condition
                if he_gamma and len(sel_neg_he)>0: ### Be careful with the meaning of this condition
                    continue
                else:
                    for num_ar, area in enumerate(areas1):
                        if max_charge_s_id in area:
                            chargs_phot  [num_ar].append(sum(qs))
                            truepos_phot [num_ar].append(phot_neg_pos[0])
                            id_max       [num_ar].append(max_charge_s_id)
                            touched_sipms[num_ar].append(len(qs))
                            evt_ids              .append(evt)
                            num_tile             .append(num_ar)

            elif len(sel_pos_phot)>0: ### Be careful with the meaning of this condition
                if he_gamma and len(sel_pos_he)>0: ### Be careful with the meaning of this condition
                    continue
                else:
                    if max_charge_s_id in area10:
                        chargs_phot  [-1].append(sum(qs))
                        truepos_phot [-1].append(phot_neg_pos[0])
                        id_max       [-1].append(max_charge_s_id)
                        touched_sipms[-1].append(len(qs))
                        evt_ids              .append(evt)
                        num_tile             .append(4)
        else:
            continue

chargs_phot   = np.array(chargs_phot  )
truepos_phot  = np.array(truepos_phot )
id_max        = np.array(id_max       )
touched_sipms = np.array(touched_sipms)
evt_ids       = np.array(evt_ids      )
num_tile      = np.array(num_tile     )

np.savez(evt_file, chargs_phot=chargs_phot, truepos_phot=truepos_phot, id_max=id_max, touched_sipms=touched_sipms, evt_ids=evt_ids, num_tile=num_tile)

print(datetime.datetime.now())
