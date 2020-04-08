import os
import sys
import argparse
import numpy    as np

from antea.io.mc_io import load_configuration

"""
Example of calling this script:

$ python analysis_coincidences_sensitivity_full_body_PET.py 0 1 /Users/carmenromoluque/Desktop/ /Users/carmenromoluque/Desktop/ full_body_iradius380mm_z200cm_depth3cm_pitch7mm
"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'    , type = int, help = "first file (inclusive)"     )
    parser.add_argument('n_files'       , type = int, help = "number of files to analize" )
    parser.add_argument('npz_files_path',             help = "input npz files path"       )
    parser.add_argument('h5_files_path' ,             help = "input h5 files path"        )
    parser.add_argument('h5_filename'   ,             help = "input h5 files name"        )
    return parser.parse_args()

arguments      = parse_args(sys.argv)
start          = arguments.first_file
numb           = arguments.n_files
npz_files_path = arguments.npz_files_path
h5_files_path  = arguments.h5_files_path
h5_filename    = arguments.h5_filename

pos_cart1_time = []
pos_cart2_time = []
event_ids      = []
num_files1     = 0
num_files2     = 0
saved_evts     = []
num_evts       = []

for filename in os.listdir(npz_files_path):
    if filename.endswith('.npz'):
        my_file = npz_files_path+filename
        print(f'Analizing file {filename}')
        d = np.load(my_file)
        num_files1 += 1

        for i in d['pos_cart1_time']:
            pos_cart1_time.append(i)
        for i in d['pos_cart2_time']:
            pos_cart2_time.append(i)
        for i in d['event_ids']:
            event_ids     .append(i)

a_pos_cart1_time = np.array(pos_cart1_time)
a_pos_cart2_time = np.array(pos_cart2_time)
a_event_ids      = np.array(event_ids)

print(len(a_pos_cart1_time), len(a_pos_cart2_time), len(a_event_ids))


for number in range(start, start+numb):
    number_str = "{:03d}".format(number)
    filename  = f"{h5_files_path}/{h5_filename}.{number_str}.pet.h5"
    try:
        h5config = load_configuration(filename)
        print(f'Analizing file {filename}')
        num_evts  .append(int(h5config[h5config.param_key=='num_events'  ].param_value.values[0]))
        saved_evts.append(int(h5config[h5config.param_key=='saved_events'].param_value.values[0]))
        num_files2 += 1
    except ValueError:
        print(f'File {filename} not found')
        continue
    except OSError:
        print(f'File {filename} not found')
        continue
    except KeyError:
        print(f'No object named MC/configuration in file {filename}')
        continue


print('')
num_npz_files          = num_files1*5
num_h5_files           = num_files2
total_events_generated = sum(num_evts)
total_saved_events     = sum(saved_evts)
total_coincidences     = len(a_pos_cart1_time)
print('Total events generated: ', total_events_generated)
print('Total saved events:     ', total_saved_events)
print('Total coincidences:     ', total_coincidences)
print('')
print('Total analized npz files: ', num_npz_files)
print('Total analized h5  files: ', num_h5_files )
print('')
