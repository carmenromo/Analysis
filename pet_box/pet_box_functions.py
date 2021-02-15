import argparse
import datetime
import numpy    as np

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('thr_ch_start' , type = int, help = "init threshold in charge"  )
    parser.add_argument('thr_ch_nsteps', type = int, help = "numb steps thrs in charge" )
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()


def info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns):
    sipms       = DataSiPM_idx.loc[evt_sns.sensor_id]
    sns_ids     = sipms.index.astype('int64').values
    sns_pos     = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges = evt_sns.charge
    sel         = sipms.Z.values<0 #Plane with 4 dices
    return sns_ids[sel], sns_pos[sel], sns_charges[sel]


def select_phot_pet_box(evt_parts, evt_hits):
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]
    ids          = sel_vol_name.particle_id.values

    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies = energies.reset_index()

    sel1 = rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)
    sel2 = rf.lower_or_equal  (energies.energy, 0.48,     allowed_error=1.e-6)
    energy_sel = energies[sel1]# & sel2]

    sel_vol_name_e  = sel_vol_name[sel_vol_name.particle_id.isin(energy_sel.particle_id)]

    primaries = evt_parts[evt_parts.primary == True]
    sel_all   = sel_vol_name_e[sel_vol_name_e.mother_id.isin(primaries.particle_id.values)]
    if len(sel_all) == 0:
        return (False, [])

    ### Once the event has passed the selection, let's calculate the true position(s)
    ids      = sel_all.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)
    sel_hits = sel_hits.groupby(['particle_id'])

    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        ave_hit_pos   = np.average(hit_positions, axis=0, weights=df.energy)
        if ave_hit_pos[2] > 0: continue
        true_pos.append(ave_hit_pos)
    if len(true_pos): return (True, true_pos[0])
    else: return (False, [])
