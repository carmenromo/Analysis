from __future__ import print_function

import numpy as np
import argparse
import sys, os
from glob import glob
import tables


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path' ,   help = "input files path" )
    parser.add_argument('filename'   ,   help = "input files name" )
    parser.add_argument('output_path',   help = "output files path")
    parser.add_argument('out_filename',  help = "output file name" )
    return parser.parse_args()

arguments     = parse_args(sys.argv)
input_path    = arguments.input_path
filename      = arguments.filename
output_path   = arguments.output_path
out_filename  = arguments.out_filename

# input/output files
files = glob(input_path + f'/{filename}.*.h5')
files = sorted(files, key=lambda s: int(s.split('.')[-2]))
out_file = f"{input_path}/{out_filename}.h5"

# Open the final tables file.
fcombined = tables.open_file(f"{output_path}/{out_filename}.h5", "w", filters=tables.Filters(complib="blosc", complevel=9))
group_reco = fcombined.create_group(fcombined.root, "reco")

# Open the first file.
f1 = tables.open_file(files[0], 'r')
reco_combined = f1.copy_node('/reco', name='table', newparent=group_reco)
f1.close()

# Process nruns files.
for fname in files[1:]:
    print(f"-- Adding file {fname}")

    # Open the next file and extract the elements.
    fn = tables.open_file(fname, 'r')
    if("/reco" in fn):
        reco_table = fn.root.reco.table
        reco_combined.append(reco_table.read())
    fn.close()

    # Flush the combined file.
    fcombined.flush()

# Close the combined hdf5 file.
print(f"Saving combined file {output_path}/{out_filename}.h5")
fcombined.close()
