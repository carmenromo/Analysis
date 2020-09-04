import pandas as pd
import numpy  as np

import antea.reco.mctrue_functions as mcf
import antea.reco.reco_functions   as rf

def true_photoelect(evt_parts, evt_hits):
    """Returns the position of the true photoelectric energy deposition
    calculated with barycenter algorithm.
    """

    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]
    ids          = sel_vol_name.particle_id.values

    sel_hits   = mcf.find_hits_of_given_particles(ids, evt_hits)
    energies   = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies   = energies.reset_index()
    energy_sel = energies[rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]

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
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    ### Reject events where the two gammas have interacted in the same hemisphere.
    #if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.511):
    #   return (False, [])

    return (True, true_pos)


def true_photoelect_new_nexus(evt_parts, evt_hits):
    """Returns the position of the true photoelectric energy deposition
    calculated with barycenter algorithm.
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
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    ### Reject events where the two gammas have interacted in the same hemisphere.
    #if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.511):
    #   return (False, [])

    return (True, true_pos)


def find_first_interactions_in_active(particles, hits, photo_range=1.):
    """
    Looks for the first interaction of primary gammas in the active volume.
    """
    ### select electrons, primary gammas daughters in ACTIVE
    sel_volume   = (particles.initial_volume == 'ACTIVE') & (particles.final_volume == 'ACTIVE')
    sel_name     = particles.name == 'e-'
    sel_vol_name = particles[sel_volume & sel_name]
    primaries = particles[particles.primary == True]
    sel_all   = sel_vol_name[sel_vol_name.mother_id.isin(primaries.particle_id.values)]
    ### Calculate the initial vertex.
    gamma_pos1, gamma_pos2 = [], []
    vol1      , vol2       = [], []
    min_t1    , min_t2     = float('inf'), float('inf')
    if len(sel_all[sel_all.mother_id == 1]) > 0:
        gamma_pos1, min_t1, _ = rf.initial_coord_first_daughter(sel_all, 1)

    if len(sel_all[sel_all.mother_id == 2]) > 0:
        gamma_pos2, min_t2, _ = rf.initial_coord_first_daughter(sel_all, 2)

    ### Calculate the minimum time among the hits of a given primary gamma,
    ### if any.
    if len(hits[hits.particle_id == 1]) > 0:
        g_pos1, g_min_t1 = rf.part_first_hit(hits, 1)
        if g_min_t1 < min_t1:
            min_t1     = g_min_t1
            gamma_pos1 = g_pos1

    if len(hits[hits.particle_id == 2]) > 0:
        g_pos2, g_min_t2 = rf.part_first_hit(hits, 2)
        if g_min_t2 < min_t2:
            min_t2     = g_min_t2
            gamma_pos2 = g_pos2

    if not len(gamma_pos1) or not len(gamma_pos2):
        print("Cannot find two true gamma interactions for this event")
        return [], [], None, None, None, None

    ## find if the event is photoelectric-like

    distances1 = rf.find_hit_distances_from_true_pos(hits, gamma_pos1)
    if max(distances1) > photo_range: ## hits at <1 mm distance are considered of the same point
        phot_like1 = False
    else:
        phot_like1 = True

    distances2 = rf.find_hit_distances_from_true_pos(hits, gamma_pos2)
    if max(distances2) > photo_range: ## hits at <1 mm distance are considered of the same point
        phot_like2 = False
    else:
        phot_like2 = True

    return gamma_pos1, gamma_pos2, min_t1, min_t2, phot_like1, phot_like2


def reconstruct_coincidences(sns_response, charge_range, DataSiPM_idx, particles, hits):
    """
    Finds the SiPM with maximum charge. Divide the SiPMs in two groups,
    separated by the plane perpendicular to the line connecting this SiPM
    with the centre of the cylinder.
    The true position of the first gamma interaction in ACTIVE is also
    returned for each of the two primary gammas (labeled 1 and 2 following
    GEANT4 ids). The two SiPM groups are assigned to their correspondent
    true gamma by position.
    A range of charge is given to select singles in the photoelectric peak.
    DataSiPM_idx is assumed to be indexed on the sensor ids. If it is not,
    it is indexed inside the function.
    """
    if 'SensorID' in DataSiPM_idx.columns:
        DataSiPM_idx = DataSiPM_idx.set_index('SensorID')

    max_sns = sns_response[sns_response.charge == sns_response.charge.max()]
    ## If by chance two sensors have the maximum charge, choose one (arbitrarily)
    if len(max_sns != 1):
        max_sns = max_sns[max_sns.sensor_id == max_sns.sensor_id.min()]
    max_sipm = DataSiPM_idx.loc[max_sns.sensor_id]
    max_pos  = np.array([max_sipm.X.values, max_sipm.Y.values, max_sipm.Z.values]).transpose()[0]

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids       = sipms.index.astype('int64').values
    sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_charges   = sns_response.charge

    sns1, sns2, pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_ids, sns_positions, sns_charges, max_pos)

    tot_q1 = sum(q1)
    tot_q2 = sum(q2)

    sel1 = (tot_q1 > charge_range[0]) & (tot_q1 < charge_range[1])
    sel2 = (tot_q2 > charge_range[0]) & (tot_q2 < charge_range[1])
    if not sel1 or not sel2:
        return [], [], [], [], None, None, None, None, [], []

    true_pos1, true_pos2, true_t1, true_t2, _, _ = find_first_interactions_in_active(particles, hits)

    if not len(true_pos1) or not len(true_pos2):
        print("Cannot find two true gamma interactions for this event")
        return [], [], [], [], None, None, None, None, [], []

    scalar_prod = true_pos1.dot(max_pos)
    if scalar_prod > 0:
        int_pos1 = pos1
        int_pos2 = pos2
        int_q1   = q1
        int_q2   = q2
        int_sns1 = sns1
        int_sns2 = sns2
    else:
        int_pos1 = pos2
        int_pos2 = pos1
        int_q1   = q2
        int_q2   = q1
        int_sns1 = sns2
        int_sns2 = sns1

    return int_pos1, int_pos2, int_q1, int_q2, true_pos1, true_pos2, true_t1, true_t2, int_sns1, int_sns2
