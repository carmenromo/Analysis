import sys
import argparse
import numpy  as np
import pandas as pd

import antea.reco.mctrue_functions as mcf

from antea.io.mc_io import load_mchits
from antea.io.mc_io import load_mcparticles


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"     )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize" )
    parser.add_argument('events_path',             help = "input files path"           )
    parser.add_argument('file_name'  ,             help = "name of input files"        )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()


arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path

evt_file  = f"{data_path}/full_body_evts_{start}_{numb}"

all_evts             = []
all_saved_evts       = []
all_single_phot_evts = []
all_coinc_phot_evts  = []


for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    #filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    filename  = f"{eventsPath}/{file_name}.{number}.h5"
    try:
        h5conf = pd.read_hdf(filename, 'MC/configuration')
    except ValueError:
        print(f'File {filename} not found')
        continue
    except OSError:
        print(f'File {filename} not found')
        continue
    except KeyError:
        print(f'No object named MC/configuration in file {filename}')
        continue
    print(f'Analyzing file {filename}')

    num_evts   = int(h5conf[h5conf.param_key=='num_events']  .param_value.values[0])
    saved_evts = int(h5conf[h5conf.param_key=='saved_events'].param_value.values[0])

    particles = load_mcparticles(filename)
    hits      = load_mchits     (filename)
    events    = particles.event_id.unique()

    single_phot = 0
    coinc_phot  = 0
    for evt in events:
        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits) #rf2.true_photoelect(evt_parts, evt_hits)
        if not select: continue
        if len(true_pos)==1:
            single_phot += 1
        elif len(true_pos)==2:
            coinc_phot += 1
        else:
            continue

    all_single_phot_evts.append(single_phot)
    all_coinc_phot_evts .append(coinc_phot)
    all_evts            .append(num_evts)
    all_saved_evts      .append(saved_evts)

all_evts             = np.array(all_evts)
all_saved_evts       = np.array(all_saved_evts)
all_single_phot_evts = np.array(all_single_phot_evts)
all_coinc_phot_evts  = np.array(all_coinc_phot_evts)

np.savez(evt_file, all_evts=all_evts, all_saved_evts=all_saved_evts, all_single_phot_evts=all_single_phot_evts, all_coinc_phot_evts=all_coinc_phot_evts)
