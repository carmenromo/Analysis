import sys
import argparse
import pandas   as pd

import antea.database.load_db  as db
import antea.mcsim   .errmat   as errmat
import antea.mcsim   .errmat3d as errmat3d
import antea.mcsim   .fastmc3d as fmc
import antea.io      .mc_io    as mcio

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize")
    parser.add_argument('input_path' ,             help = "input files path"          )
    parser.add_argument('filename'   ,             help = "input files name"          )
    parser.add_argument('output_path',             help = "output files path"         )
    parser.add_argument('matrix_path',             help = "path of the error matrix"  )
    return parser.parse_args()


DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')


arguments     = parse_args(sys.argv)
start         = arguments.first_file
numb_of_files = arguments.n_files
folder_in     = arguments.input_path
filename      = arguments.filename
folder_out    = arguments.output_path
table_folder  = arguments.matrix_path


err_r_phot_file      = table_folder + '/errmat_nest_r_phot_like.npz'
err_r_compt_file     = table_folder + '/errmat_nest_r_compt_like.npz'
err_phi_phot_file    = table_folder + '/errmat_nest_phi_phot_like.npz'
err_phi_compt_file   = table_folder + '/errmat_nest_phi_compt_like.npz'
err_z_phot_file      = table_folder + '/errmat_nest_z_phot_like.npz'
err_z_compt_file     = table_folder + '/errmat_nest_z_compt_like.npz'
err_t_th_phot_file0  = table_folder + '/errmat_nest_t_thr0pes_phot_like.npz'
err_t_th_compt_file0 = table_folder + '/errmat_nest_t_thr0pes_compt_like.npz'
err_t_th_phot_file1  = table_folder + '/errmat_nest_t_thr0.5pes_phot_like.npz'
err_t_th_compt_file1 = table_folder + '/errmat_nest_t_thr0.5pes_compt_like.npz'
err_t_th_phot_file2  = table_folder + '/errmat_nest_t_thr1.0pes_phot_like.npz'
err_t_th_compt_file2 = table_folder + '/errmat_nest_t_thr1.0pes_compt_like.npz'
err_t_th_phot_file3  = table_folder + '/errmat_nest_t_thr1.5pes_phot_like.npz'
err_t_th_compt_file3 = table_folder + '/errmat_nest_t_thr1.5pes_compt_like.npz'

errmat_r_phot      = errmat.errmat(err_r_phot_file)
errmat_r_compt     = errmat.errmat(err_r_compt_file)
errmat_phi_phot    = errmat3d.errmat3d(err_phi_phot_file)
errmat_phi_compt   = errmat3d.errmat3d(err_phi_compt_file)
errmat_z_phot      = errmat3d.errmat3d(err_z_phot_file)
errmat_z_compt     = errmat3d.errmat3d(err_z_compt_file)
errmat_t_th_phot0  = errmat.errmat(err_t_th_phot_file0)
errmat_t_th_compt0 = errmat.errmat(err_t_th_compt_file0)
errmat_t_th_phot1  = errmat.errmat(err_t_th_phot_file1)
errmat_t_th_compt1 = errmat.errmat(err_t_th_compt_file1)
errmat_t_th_phot2  = errmat.errmat(err_t_th_phot_file2)
errmat_t_th_compt2 = errmat.errmat(err_t_th_compt_file2)
errmat_t_th_phot3  = errmat.errmat(err_t_th_phot_file3)
errmat_t_th_compt3 = errmat.errmat(err_t_th_compt_file3)


for file_number in range(start, start+numb_of_files):
    sim_file  = folder_in  + '/' + filename + f'.{file_number}.pet.h5'
    out_file0 = folder_out + '/' + filename + f'_reco_thr0pes.{file_number}.h5'
    out_file1 = folder_out + '/' + filename + f'_reco_thr0.5pes.{file_number}.h5'
    out_file2 = folder_out + '/' + filename + f'_reco_thr1.0pes.{file_number}.h5'
    out_file3 = folder_out + '/' + filename + f'_reco_thr1.5pes.{file_number}.h5'

    try:
        particles = mcio.load_mcparticles(sim_file)
    except:
        print(f'File {sim_file} not found!')
        continue
    hits   = mcio.load_mchits(sim_file)
    events = particles.event_id.unique()

    reco0 = pd.DataFrame(columns=['event_id', 'true_energy',
                                  'true_r1', 'true_phi1', 'true_z1', 'true_t1',
                                  'true_r2', 'true_phi2', 'true_z2', 'true_t2',
                                  'phot_like1', 'phot_like2',
                                  'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
                                  'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2'])

    reco1 = pd.DataFrame(columns=['event_id', 'true_energy',
                                  'true_r1', 'true_phi1', 'true_z1', 'true_t1',
                                  'true_r2', 'true_phi2', 'true_z2', 'true_t2',
                                  'phot_like1', 'phot_like2',
                                  'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
                                  'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2'])

    reco2 = pd.DataFrame(columns=['event_id', 'true_energy',
                                  'true_r1', 'true_phi1', 'true_z1', 'true_t1',
                                  'true_r2', 'true_phi2', 'true_z2', 'true_t2',
                                  'phot_like1', 'phot_like2',
                                  'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
                                  'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2'])

    reco3 = pd.DataFrame(columns=['event_id', 'true_energy',
                                  'true_r1', 'true_phi1', 'true_z1', 'true_t1',
                                  'true_r2', 'true_phi2', 'true_z2', 'true_t2',
                                  'phot_like1', 'phot_like2',
                                  'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
                                  'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2'])

    for evt in events:

        evt_df0 = fmc.simulate_reco_event(evt, hits, particles, errmat_r_phot,
                                          errmat_phi_phot, errmat_z_phot, errmat_t_th_phot0, errmat_r_compt,
                                          errmat_phi_compt, errmat_z_compt, errmat_t_th_compt0, 0.9)
        reco0 = pd.concat([reco0, evt_df0])

        evt_df1 = fmc.simulate_reco_event(evt, hits, particles, errmat_r_phot,
                                          errmat_phi_phot, errmat_z_phot, errmat_t_th_phot1, errmat_r_compt,
                                          errmat_phi_compt, errmat_z_compt, errmat_t_th_compt1, 0.9)
        reco1 = pd.concat([reco1, evt_df1])

        evt_df2 = fmc.simulate_reco_event(evt, hits, particles, errmat_r_phot,
                                          errmat_phi_phot, errmat_z_phot, errmat_t_th_phot2, errmat_r_compt,
                                          errmat_phi_compt, errmat_z_compt, errmat_t_th_compt2, 0.9)
        reco2 = pd.concat([reco2, evt_df2])

        evt_df3 = fmc.simulate_reco_event(evt, hits, particles, errmat_r_phot,
                                          errmat_phi_phot, errmat_z_phot, errmat_t_th_phot3, errmat_r_compt,
                                          errmat_phi_compt, errmat_z_compt, errmat_t_th_compt3, 0.9)
        reco3 = pd.concat([reco3, evt_df3])


    store = pd.HDFStore(out_file0, "w", complib=str("zlib"), complevel=4)
    store.put('reco', reco0, format='table', data_columns=True)
    store.close()

    store = pd.HDFStore(out_file1, "w", complib=str("zlib"), complevel=4)
    store.put('reco', reco1, format='table', data_columns=True)
    store.close()

    store = pd.HDFStore(out_file2, "w", complib=str("zlib"), complevel=4)
    store.put('reco', reco2, format='table', data_columns=True)
    store.close()

    store = pd.HDFStore(out_file2, "w", complib=str("zlib"), complevel=4)
    store.put('reco', reco3, format='table', data_columns=True)
    store.close()
