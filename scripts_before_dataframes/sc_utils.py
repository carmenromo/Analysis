import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize")
    parser.add_argument('n_steps'    , type = int, help = "number of steps of thr"    )
    parser.add_argument('thr_start'  , type = int, help = "initial thr"               )
    parser.add_argument('events_path',             help = "input files path"          )
    parser.add_argument('file_name'  ,             help = "name of input files"       )
    parser.add_argument('data_path'  ,             help = "output files path"         )
    return parser.parse_args()
