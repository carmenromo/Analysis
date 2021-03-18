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

def parse_args_no_ths_and_zpos(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file'   , type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'      , type = int, help = "number of files to analize")
    parser.add_argument('in_path'      ,             help = "input files path"          )
    parser.add_argument('file_name'    ,             help = "name of input files"       )
    parser.add_argument('zpos_file'    ,             help = "Zpos table"                )
    parser.add_argument('out_path'     ,             help = "output files path"         )
    return parser.parse_args()

def info_from_sensors_with_neg_z(DataSiPM_idx, evt_sns):
    sipms       = DataSiPM_idx.loc[evt_sns.sensor_id]
    sns_ids     = sipms.index.astype('int64').values
    sns_pos     = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges = evt_sns.charge.values
    sel         = sipms.Z.values<0 #Plane with 4 dices
    return sns_ids[sel], sns_pos[sel], sns_charges[sel]

def info_from_sensors_with_pos_z(DataSiPM_idx, evt_sns):
    sipms       = DataSiPM_idx.loc[evt_sns.sensor_id]
    sns_ids     = sipms.index.astype('int64').values
    sns_pos     = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges = evt_sns.charge.values
    sel         = sipms.Z.values>0 #Plane with 1 dices
    return sns_ids[sel], sns_pos[sel], sns_charges[sel]

def info_from_sensors_for_a_given_tile(DataSiPM_idx, evt_sns, sns_ids_tile):
    sel_sns     = evt_sns.sensor_id.isin(sns_ids_tile)
    sipms       = DataSiPM_idx.loc[evt_sns[sel_sns].sensor_id]
    sns_ids     = sipms.index.astype('int64').values
    sns_pos     = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges = evt_sns[evt_sns.sensor_id.isin(sns_ids_tile)].charge.values
    return sns_ids, sns_pos, sns_charges


def select_gamma_high_energy(evt_parts, evt_hits):
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]

    gamma_he  = evt_parts[evt_parts.kin_energy == 1.274537]
    sel_mother = sel_vol_name[sel_vol_name.mother_id.isin(gamma_he.particle_id.values)]

    ids      = sel_mother.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies = energies.reset_index()

    energy_sel = energies[energies.energy == energies.energy.max()]
    sel_all = sel_mother[sel_mother.particle_id.isin(energy_sel.particle_id)]

    if len(sel_all)==0:
        return (False, [])

    ### Once the event has passed the selection, let's calculate the true position(s)
    ids      = sel_all.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)

    sel_hits = sel_hits.groupby(['particle_id'])
    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    return (True, np.array(true_pos))


def select_photoelectric_pet_box(evt_parts, evt_hits):
    """
    Select only the events where one or two photoelectric events occur, and nothing else.
    """
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]
    ids          = sel_vol_name.particle_id.values

    sel_hits   = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies   = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies   = energies.reset_index()
    energy_sel = energies[rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]

    sel_vol_name_e  = sel_vol_name[sel_vol_name.particle_id.isin(energy_sel.particle_id)]

    primaries = evt_parts[(evt_parts.primary == True) & (evt_parts.kin_energy == 0.510999)]
    sel_all   = sel_vol_name_e[sel_vol_name_e.mother_id.isin(primaries.particle_id.values)]

    if len(sel_all) == 0:
        return (False, np.array([]))

    ### Once the event has passed the selection, let's calculate the true position(s)
    ids      = sel_all.particle_id.values
    sel_hits = mcf.find_hits_of_given_particles(ids, evt_hits)

    sel_hits = sel_hits.groupby(['particle_id'])
    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    return (True, np.array(true_pos))
