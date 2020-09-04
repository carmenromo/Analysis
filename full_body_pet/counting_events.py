import sys
import math
import argparse
import numpy    as np

from antea.io.mc_io import load_configuration

"""
python counting_events.py 22 1 /Users/carmenromoluque/Desktop/ full_body_iradius380mm_z200cm_depth4cm_pitch7mm /Users/carmenromoluque/Desktop/
"""

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

saved_events = []
wrong_files  = []

evt_file = f"{data_path}/full_body_counting_evts_{start}_{numb}"

for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    try:
        h5conf     = load_configuration(filename)
        saved_evts = int(h5conf[h5conf.param_key=='saved_events'].param_value.values[0])
        saved_events.append(saved_evts)
    except FileNotFoundError:
        #print(f'File {filename} is not ok or does not exist')
        wrong_files.append(number)
        continue
    except ValueError:
        #print('File {filename} not found')
        wrong_files.append(number)
        continue
    except OSError:
        #print('File {filename} not found')
        wrong_files.append(number)
        continue
    except KeyError:
        #print(f'No object named MC/particles in file {filename}'
        wrong_files.append(number)
        continue

np.savez(evt_file, saved_event=np.array(saved_events), wrong_files=np.array(wrong_files))
