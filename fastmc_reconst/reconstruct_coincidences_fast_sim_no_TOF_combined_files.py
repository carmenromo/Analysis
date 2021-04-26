import os
import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

import mlem.mlem_reconstruct as mr


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('file_fast'     ,             help = "input files fast"            )
    parser.add_argument('file_full'     ,             help = "input files full"            )
    parser.add_argument('threshold'     ,             help = "TOF threshold"               )
    parser.add_argument('ctr'           , type = int, help = "time resolution for this thr")
    parser.add_argument('output_im_path',             help = "output images path"          )
    parser.add_argument('path_to_mlem'  ,             help = "path to the mlem algorithm"  )
    parser.add_argument('n_iterations'  , type = int, help = "number of iterations"        )
    parser.add_argument('save_every'    , type = int, help = "save every n iterations"     )
    return parser.parse_args()

arguments = parse_args(sys.argv)
file_fast = arguments.file_fast
file_full = arguments.file_full

th            = arguments.threshold
tof           = arguments.ctr
folder_out_im = arguments.output_im_path
path_to_mlem  = arguments.path_to_mlem
n_iterations  = arguments.n_iterations
save_every    = arguments.save_every

cols = ['event_id', 'true_energy',
        'true_r1', 'true_phi1', 'true_z1', 'true_t1',
        'true_r2', 'true_phi2', 'true_z2', 'true_t2', 'phot_like1', 'phot_like2',
        'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
        'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2']

times_fast = []
times_full = []
times_fast.append(datetime.datetime.now())

table = pd.read_hdf(file_fast, 'reco/table')
sel_below_th = (table.true_energy > 0.) & (table.true_r1 == 0.)
reco_fast = table[~sel_below_th]
times_fast.append(datetime.datetime.now())


## FULLSIM FILES
times_full.append(datetime.datetime.now())
table        = pd.read_hdf(file_full, 'reco/table')
sel_below_th = (table.true_energy > 0.) & (table.true_r1 == 0.)
reco_full    = table[~sel_below_th]
times_full.append(datetime.datetime.now())

df = pd.concat([reco_fast, reco_full], ignore_index=True, sort=False)

print('times_fast: ', times_fast[0], times_fast[-1])
print('times_full: ', times_full[0], times_full[-1])

print('------------------------------------------------------------')
print('--- Number of coincidences passing the energy threshold: ---')
print('------------------------------------------------------------')
print(len(df), f' coincidences  (thr {th} pes)')
print('no CTR')
print('')

r1   = df.reco_r1  .values
phi1 = df.reco_phi1.values
z1   = df.reco_z1  .values
t1   = df.reco_t1  .values
r2   = df.reco_r2  .values
phi2 = df.reco_phi2.values
z2   = df.reco_z2  .values
t2   = df.reco_t2  .values
evt  = df.event_id .values

lor_x1 = r1*np.cos(phi1); lor_y1 = r1*np.sin(phi1); lor_z1 = z1; lor_t1 = t1;
lor_x2 = r2*np.cos(phi2); lor_y2 = r2*np.sin(phi2); lor_z2 = z2; lor_t2 = t2;


## Perform the 3D PET reconstruction:
path_to_mlem = path_to_mlem + 'libmlem.so'

rec                = mr.MLEMReconstructor(libpath=path_to_mlem)
rec.TOF            = False
#rec.TOF_resolution = tof
rec.niterations    = n_iterations
rec.save_every     = save_every
coincs = [100000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 40000000]
for nc in coincs:
    n_coincidences = nc #len(df)
    rec.prefix     = folder_out_im + f'im_th{th}_no_TOF_{n_coincidences}coinc_iter'
    img            = rec.reconstruct(lor_x1[:nc], lor_y1[:nc], lor_z1[:nc], lor_t1[:nc], lor_x2[:nc], lor_y2[:nc], lor_z2[:nc], lor_t2[:nc])

print('End of the file: ', datetime.datetime.now())
