
import argparse
import datetime

import numpy  as np
import pandas as pd

from typing import Sequence, Tuple

import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.core.exceptions import WaveformEmptyTable

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
    parser.add_argument('first_file', type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'   , type = int, help = "number of files to analize")
    parser.add_argument('in_path'   ,             help = "input files path"          )
    parser.add_argument('file_name' ,             help = "name of input files"       )
    parser.add_argument('zpos_file' ,             help = "Zpos table"                )
    parser.add_argument('out_path'  ,             help = "output files path"         )
    return parser.parse_args()

def parse_args_no_ths_and_zpos2(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file', type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'   , type = int, help = "number of files to analize")
    parser.add_argument('in_path'   ,             help = "input files path"          )
    parser.add_argument('file_name' ,             help = "name of input files"       )
    parser.add_argument('zpos_file' ,             help = "Zpos table det plane"      )
    parser.add_argument('zpos_file2',             help = "Zpos table coinc plane"    )
    parser.add_argument('out_path'  ,             help = "output files path"         )
    return parser.parse_args()


def info_from_the_tiles(DataSiPM_idx, evt_sns):
    sipms       = DataSiPM_idx.loc[evt_sns.sensor_id]
    sns_ids     = sipms.index.astype('int64').values
    sns_pos     = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges = evt_sns.charge.values
    sel = sipms.Z.values<0
    return (sns_ids[ sel], sns_pos[ sel], sns_charges[ sel], #Plane with 4 tiles
            sns_ids[~sel], sns_pos[~sel], sns_charges[~sel]) #Plane with 1 tile


def select_phot_pet_box(evt_parts: pd.DataFrame,
                        evt_hits:  pd.DataFrame,
                        he_gamma: str = False) -> Tuple[bool, Sequence[Tuple[float, float, float]]]:
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
    if he_gamma:
        energy_sel = energies[rf.greater_or_equal(energies.energy, 1.23998, allowed_error=1.e-5)]
        primaries = evt_parts[(evt_parts.primary == True) & (evt_parts.kin_energy == 1.274537)]
    else:
        energy_sel = energies[rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]
        primaries = evt_parts[(evt_parts.primary == True) & (evt_parts.kin_energy == 0.510999)]

    sel_vol_name_e = sel_vol_name  [sel_vol_name  .particle_id.isin(energy_sel.particle_id)]
    sel_all        = sel_vol_name_e[sel_vol_name_e.mother_id  .isin(primaries .particle_id.values)]

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


def find_first_time_of_sensors(tof_response: pd.DataFrame,
                               sns_ids: Sequence[int],
                               sigma: float = 30, n_pe: int = 1)-> Tuple[int, int]:
    """
    This function looks for the time among all sensors for the first
    photoelectron detected.
    In case more than one photoelectron arrives at the same time,
    the sensor with minimum id is chosen.
    The positive value of the id of the sensor and the time of detection
    are returned.
    """
    tof = tof_response[tof_response.sensor_id.isin(sns_ids)]
    if tof.empty:
        raise WaveformEmptyTable("Tof dataframe is empty")

    tof.insert(4, 'jit_time', np.random.normal(tof.time.values, sigma))

    first_times = tof.sort_values(by=['jit_time']).iloc[0:n_pe]
    min_t       = first_times['jit_time'].mean()
    min_ids     = first_times.sensor_id.values
    min_charges = first_times.charge.values

    return np.abs(min_ids), min_charges, min_t

def find_coincidence_timestamps(tof_response: pd.DataFrame,
                                sns1: Sequence[int],
                                sns2: Sequence[int],
                                sigma: float, npe: int)-> Tuple[int, int, int, int]:
    """
    Finds the first time and sensor of each one of two sets of sensors,
    given a sensor response dataframe.
    """
    min1, q1, time1 = find_first_time_of_sensors(tof_response, -sns1, sigma, npe)
    min2, q2, time2 = find_first_time_of_sensors(tof_response, -sns2, sigma, npe)

    return min1, min2, q1, q2, time1, time2
