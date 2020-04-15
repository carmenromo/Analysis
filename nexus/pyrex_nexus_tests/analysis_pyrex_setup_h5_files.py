import sys
import argparse
import numpy    as np

from antea.io.mc_io import load_configuration
from antea.io.mc_io import load_mcparticles


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('n_files'  , type = int, help = "number of files to analize")
    parser.add_argument('filespath', type = int, help = "Input files path"          )
    return parser.parse_args()


arguments = parse_args(sys.argv)
numb      = arguments.n_files
filespath = arguments.filespath

files     = [f'test_setup{i}.pet.h5' for i in range(numb)]
evt_file = filespath + f'output_file.npz'

list_num_evts_interact1 = []
list_num_evts_interact2 = []
list_num_evts_both      = []

for file in files:
    h5conf = load_configuration(filespath+file)
    h5part = load_mcparticles  (filespath+file)

    gen_evts   = int(h5conf[h5conf.param_key=='num_events'        ].param_value.values[0])
    saved_evts = int(h5conf[h5conf.param_key=='saved_events'      ].param_value.values[0])
    inter_evts = int(h5conf[h5conf.param_key=='interacting_events'].param_value.values[0])

    assert gen_evts   == 1000000
    assert saved_evts == 1000000
    assert inter_evts == 0

    events = h5part.event_id.unique()
    num_evts_interact1 = 0
    num_evts_interact2 = 0
    num_evts_both      = 0
    for evt in events[:]:
        evt_parts    = h5part[h5part.event_id == evt]
        sel_volume1  = (evt_parts.initial_volume == 'WALL')
        sel_volume2  = (evt_parts.final_volume   == 'WALL')

        sel_vol1 = evt_parts[sel_volume1]
        sel_vol2 = evt_parts[sel_volume2]
        sel_vol3 = evt_parts[sel_volume1 & sel_volume2]

        if len(sel_vol1)>0:
            num_evts_interact1 += 1
        if len(sel_vol2)>0:
            num_evts_interact2 += 1
        if len(sel_vol3)>0:
            num_evts_both += 1

    list_num_evts_interact1.append(num_evts_interact1)
    list_num_evts_interact2.append(num_evts_interact2)
    list_num_evts_both     .append(num_evts_both)

print(list_num_evts_interact1)
print(list_num_evts_interact2)

np.savez(evt_file, num_evts_interact1=list_num_evts_interact1, num_evts_interact2=list_num_evts_interact2, num_evts_both=list_num_evts_both)
