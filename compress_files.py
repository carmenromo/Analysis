import os
import sys
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, nargs="*", help="input files"       )
    parser.add_argument( "-c"        , action="store_true", help="Compression mode"  )
    parser.add_argument( "-d"        , action="store_true", help="Decompression mode")
    return parser.parse_args()

arguments   = parse_args(sys.argv)
input_files = arguments.input_files
compress    = arguments.c
decompress  = arguments.d

for ifile in input_files:
    if compress and not decompress:
        print(f"Compressing file {ifile}")
        os.system(f"gzip {ifile}")
    elif decompress and not compress:
        print(f"Decompressing file {ifile}")
        os.system(f"gzip -d {ifile}")
    else:
        print("Incorrect commands")
        continue
