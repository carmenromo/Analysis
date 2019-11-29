import numpy  as np
import pandas as pd

from typing import Sequence, Tuple
import antea.reco.reco_functions as rf

#@profile
def from_cartesian_to_cyl(pos: Sequence[np.array]) -> Sequence[np.array]:
    cyl_pos = np.array([np.sqrt(pos[:,0]**2+pos[:,1]**2), np.arctan2(pos[:,1], pos[:,0]), pos[:,2]]).transpose()
    return cyl_pos


#@profile
def find_first_time_of_sensors(tof_response: pd.DataFrame, sns_ids: Sequence[int])-> Tuple[int, float]:
    """
    This function looks for the time among all sensors for the first photoelectron detected.
    In case more than one photoelectron arrives at the same time, the sensor with minimum id is chosen.
    The positive value of the id of the sensor and the time of detection are returned.
    """
    tof    = tof_response[tof_response.sensor_id.isin(sns_ids)]
    min_t  = tof.in_time.min()
    min_df = tof[tof.in_time == min_t]

    #if len(min_df)==0:
    #    return None, None
    if len(min_df)>1:
        min_id = min_df[min_df.sensor_id == min_df.sensor_id.min()].sensor_id.values[0]
    else:
        min_id = min_df.sensor_id.values[0]

    return min_id, min_t


#@profile
def reconstruct_coincidences(sns_response: pd.DataFrame, charge_range: Tuple[float, float], DataSiPM_idx: pd.DataFrame):
    max_sns       = sns_response[sns_response.data == sns_response.data.max()]
    ## If by chance two sensors have the maximum charge, choose one (arbitrarily)
    if len(max_sns != 1):
        max_sns   = max_sns[max_sns.sensor_id == max_sns.sensor_id.min()]
    max_sipm      = DataSiPM_idx.loc[max_sns.sensor_id]
    max_pos       = np.array([max_sipm.X.values, max_sipm.Y.values, max_sipm.Z.values]).transpose()[0]
    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = sns_response.data

    sns1, sns2, pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_ids, sns_positions, sns_charges, max_pos)

    tot_q1 = sum(q1)
    tot_q2 = sum(q2)
    sel1 = (tot_q1 > charge_range[0]) & (tot_q1 < charge_range[1])
    sel2 = (tot_q2 > charge_range[0]) & (tot_q2 < charge_range[1])
    if not sel1 or not sel2:
        return [], [], [], [], [], [], None, None, None, None
    
    ### TOF
    min1, min_tof1 = find_first_time_of_sensors(sns_response, sns1)
    min2, min_tof2 = find_first_time_of_sensors(sns_response, sns2)
    
    return sns1, sns2, pos1, pos2, q1, q2, min1, min2, min_tof1, min_tof2


#@profile
def divide_sipms_in_two_hemispheres(sns_ids: Sequence[int], sns_positions: Sequence[Tuple[float, float, float]], sns_charges: Sequence[float], reference_pos: Tuple[float, float, float]) -> Tuple[Sequence[int], Sequence[int], Sequence[float], Sequence[float], Sequence[Tuple[float, float, float]], Sequence[Tuple[float, float, float]]]:
    """
    Divide the SiPMs with charge between two hemispheres, using a given reference direction
    (reference_pos) as a discriminator.
    Return the lists of the ids, the charges and the positions of the SiPMs of the two groups.
    """

    q1,   q2   = [], []
    pos1, pos2 = [], []
    id1, id2   = [], []
    for sns_id, sns_pos, charge in zip(sns_ids, sns_positions, sns_charges):
        scalar_prod = sns_pos.dot(reference_pos)
        if scalar_prod > 0.:
            q1  .append(charge)
            pos1.append(sns_pos)
            id1.append(sns_id)
        else:
            q2  .append(charge)
            pos2.append(sns_pos)
            id2.append(sns_id)

    return id1, id2, pos1, pos2, np.array(q1), np.array(q2)


#@profile
def get_var_phi(posr, qr):
    pos_phi = from_cartesian_to_cyl(np.array(posr))[:,1]
    diff_sign = min(pos_phi) < 0 < max(pos_phi)
    if diff_sign & (np.abs(np.min(pos_phi))>np.pi/2.):
        pos_phi[pos_phi<0] = np.pi + np.pi + pos_phi[pos_phi<0]
    mean_phi = np.average(pos_phi, weights=qr)
    var_phi  = np.average((pos_phi-mean_phi)**2, weights=qr)
    return var_phi


#@profile
def threshold_filter(thr, q, pos, sns):
    sel      = q > thr
    q_filt   = q  [sel]
    pos_filt = pos[sel]
    sns_filt = sns[sel]
    return q_filt, pos_filt, sns_filt
