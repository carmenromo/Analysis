import os
import sys
import argparse
import antea.io.mc_io as mcio

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize")
    parser.add_argument('input_path' ,             help = "input files path"          )
    parser.add_argument('filename'   ,             help = "input files name"          )
    parser.add_argument('output_path',             help = "output files path"         )
    return parser.parse_args()

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb_of_files = arguments.n_files
folder_in     = arguments.input_path
filename      = arguments.filename
folder_out    = arguments.output_path

for file_number in range(start, start+numb_of_files):
    sim_file     = folder_in  + '/' + filename + f'.{file_number}.h5'
    out_file     = folder_out + '/' + filename + f'.{file_number}.csv'
    out_file_tof = folder_out + '/' + filename + f'_tof.{file_number}.csv'

    try:
        sns_response = mcio.load_mcsns_response(sim_file)
    except:
        print(f'File {sim_file} not found!')
        continue
    tof_response = mcio.load_mcTOFsns_response(sim_file)

    sns_response.to_csv(out_file,     index = False)
    tof_response.to_csv(out_file_tof, index = False)
