import os
import sys
import argparse
import datetime
import numpy  as np
import pandas as pd

import mlem.mlem_reconstruct as mr


"""
Example of calling this script:
python reconstruct_coincidences_fast_sim_only_phot_1mm.py 10000 1 /Users/carmenromoluque/nexus_petit_analysis/fastmc_reconst/ full_body_phantom_paper_true_reco_thr
                                                          0 0 /Users/carmenromoluque/nexus_petit_analysis/full-body-phantom-paper/fastsim_from_full/ full_body_phantom_paper_reco_thr
                                                          0.25 140 /Users/carmenromoluque/nexus_petit_analysis/full-body-phantom-paper/images_reco/ /Users/carmenromoluque/tofpet3d/lib/ 16 2
"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'     , type = int, help = "first file (inclusive)"      )
    parser.add_argument('n_files'        , type = int, help = "number of files to analize"  )
    parser.add_argument('input_path'     ,             help = "input files path"            )
    parser.add_argument('filename'       ,             help = "input files name"            )
    parser.add_argument('first_file1'    , type = int, help = "first file (inclusive)"      )
    parser.add_argument('n_files1'       , type = int, help = "number of files to analize"  )
    parser.add_argument('input_path1'    ,             help = "input files path"            )
    parser.add_argument('filename1'      ,             help = "input files name"            )
    parser.add_argument('first_file_full', type = int, help = "first file (inclusive) from fullsim"      )
    parser.add_argument('n_files_full'   , type = int, help = "number of files to analize from fullsim"  )
    parser.add_argument('input_path_full',             help = "input fullsim files path"            )
    parser.add_argument('filename_full'  ,             help = "input fullsim files name"            )
    parser.add_argument('threshold'      ,             help = "TOF threshold"               )
    parser.add_argument('ctr'            , type = int, help = "time resolution for this thr")
    parser.add_argument('output_im_path' ,             help = "output images path"          )
    parser.add_argument('path_to_mlem'   ,             help = "path to the mlem algorithm"  )
    parser.add_argument('n_iterations'   , type = int, help = "number of iterations"        )
    parser.add_argument('save_every'     , type = int, help = "save every n iterations"     )
    return parser.parse_args()

arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb_of_files = arguments.n_files
folder_in     = arguments.input_path
filename      = arguments.filename

start1         = arguments.first_file1
numb_of_files1 = arguments.n_files1
folder_in1     = arguments.input_path1
filename1      = arguments.filename1

start_full         = arguments.first_file_full
numb_of_files_full = arguments.n_files_full
folder_in_full     = arguments.input_path_full
filename_full      = arguments.filename_full

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

df = pd.DataFrame(columns=cols)

times_fast1 = []
times_fast2 = []
times_full  = []


##Paola fastsim files
times_fast1.append(datetime.datetime.now())
for file_number in range(start, start+numb_of_files):
    fnumb = int(str(file_number)[0:-3])
    file_name = folder_in + f'files{fnumb}000/' + filename + f'0_25pes.{file_number}.h5'
    #print(file_number)
    try:
        table = pd.read_hdf(file_name, 'reco/table')
    except FileNotFoundError:
        print(f'File {file_name} not found')
        continue
    times_fast1.append(datetime.datetime.now())
    sel_below_th  = (table.true_energy > 0.) & (table.true_r1 == 0.)
    sel_phot_evts = (table.phot_like1 == 1.) & (table.phot_like2 == 1.)
    reco = table[~sel_below_th & sel_phot_evts]
    df = pd.concat([df, reco], ignore_index=True, sort=False)

print('Coinc from Paola s fastsim files: ', len(df))


##My fastsim files
times_fast2.append(datetime.datetime.now())
for file_number1 in range(start1, start1+numb_of_files1):
    file_name1 = folder_in1 + filename1 + f'{th}pes.{file_number1}.h5'
    #print(file_number1)
    try:
        table = pd.read_hdf(file_name1, 'reco/table')
    except FileNotFoundError:
        print(f'File {file_name1} not found')
        continue
    times_fast2.append(datetime.datetime.now())
    sel_below_th  = (table.true_energy > 0.) & (table.true_r1 == 0.)
    sel_phot_evts = (table.phot_like1 == 1.) & (table.phot_like2 == 1.)
    reco = table[~sel_below_th & sel_phot_evts]
    df = pd.concat([df, reco], ignore_index=True, sort=False)

print('Coinc from my fastsim files + Paola s: ', len(df))



## FULLSIM FILES
times_full.append(datetime.datetime.now())
for file_number_full in range(start_full, start_full+numb_of_files_full):
    file_name_full = folder_in_full + filename_full + f'{th}pes.{file_number_full}.h5'
    #print(file_number_full)
    try:
        table = pd.read_hdf(file_name_full, 'reco/table')
    except FileNotFoundError:
        print(f'File {file_name_full} not found')
        continue
    times_full.append(datetime.datetime.now())
    sel_below_th  = (table.true_energy > 0.) & (table.true_r1 == 0.)
    sel_phot_evts = (table.phot_like1 == 1.) & (table.phot_like2 == 1.)
    reco = table[~sel_below_th & sel_phot_evts]
    df = pd.concat([df, reco], ignore_index=True, sort=False)


print('times_fast paola: ',  times_fast1[0], times_fast1[-1])
print('times_fast carmen: ', times_fast2[0], times_fast2[-1])
print('times_full: ', times_full[0], times_full[-1])

print('------------------------------------------------------------')
print('--- Number of coincidences passing the energy threshold: ---')
print('------------------------------------------------------------')
print(len(df), f' coincidences  (thr TOF {th} pes)')
print('CTR corresponding to this threshold: ', tof)
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
#tof = [140, 220, 240] #Time resolution in ps for each threshold case

rec                = mr.MLEMReconstructor(libpath=path_to_mlem)
rec.TOF            = True
rec.TOF_resolution = tof
rec.niterations    = n_iterations
rec.save_every     = save_every
n_coincidences     = len(df)
rec.prefix         = folder_out_im + f'im_th{th}_TOF{tof}ps_phot_like_phot_range1mm_{n_coincidences}coinc_iter'
img                = rec.reconstruct(lor_x1, lor_y1, lor_z1, lor_t1, lor_x2, lor_y2, lor_z2, lor_t2)

print('End of the file: ', datetime.datetime.now())
