import os
import argparse
import tables as tb
import numpy  as np

from   invisible_cities.io.mcinfo_io  import read_mcinfo

from invisible_cities.core.exceptions import SipmEmptyList
from invisible_cities.core.exceptions import SipmZeroCharge
from invisible_cities.core.exceptions import NoHits

from invisible_cities.reco.corrections import Correction
from invisible_cities.io.dst_io        import load_dst


def sensor_position(h5in):
    """A dictionary that stores the position of all the sensors
    in cartesian coordinates is created
    """
    sipms    = h5in.root.MC.sensor_positions[:]
    sens_pos = {}
    for sipm in sipms:
        sens_pos[sipm[0]] = (sipm[1], sipm[2], sipm[3])
    return sens_pos


def sensor_position_cyl(h5in):
    """A dictionary that stores the position of all the sensors
    in cylindrical coordinates is created
    """
    sipms        = h5in.root.MC.sensor_positions[:]
    sens_pos_cyl = {}
    for sipm in sipms:
        sens_pos_cyl[sipm[0]] = (np.sqrt(sipm[1]*sipm[1] + sipm[2]*sipm[2]), np.arctan2(sipm[2], sipm[1]), sipm[3])
    return sens_pos_cyl


def find_closest_sipm(x, y, z, sens_pos, sns_over_thr, charges_over_thr):
    """Returns the sensor ID of the closest sipm to the given true event
    """
    min_dist = 1.e9
    min_sns  = 0
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos  = sens_pos[sns_id]
        dist = np.linalg.norm(np.subtract((x, y, z), pos))
        if dist < min_dist:
            min_dist = dist
            min_sns  = sns_id
    return min_sns


def barycenter_3D(pos, qs):
    """Returns the weighted position of an array
    """
    if not len(pos)   : raise SipmEmptyList
    if np.sum(qs) == 0: raise SipmZeroCharge

    return np.average(pos, weights=qs, axis=0)


def get_r_and_var_phi(ave_true, cyl_pos, q):
    r        = np.sqrt(ave_true[0]**2 + ave_true[1]**2)
    phi_pos  = np.array([el[1] for el in cyl_pos])
    diff_sign = min(phi_pos) < 0 < max(phi_pos)
    if diff_sign & (np.abs(np.min(phi_pos))>np.pi/2):
        phi_pos[phi_pos<0] = np.pi + np.pi + phi_pos[phi_pos<0]
    mean_phi = np.average(phi_pos, weights=q)
    var_phi  = np.average((phi_pos - mean_phi)**2, weights=q)
    return r, var_phi


def load_rpos(filename, group = "Radius", node = "f100bins"):
    dst = load_dst(filename, group, node)
    return Correction((dst.Sigma_phi.values,), dst.Rpos.values, dst.Uncertainty.values)


def load_zr_corrections(filename, *,
                        group = "Corrections",
                        node  = "ZRcorrections",
                        **kwargs):
    dst  = load_dst(filename, group, node)
    z, r = np.unique(dst.z.values), np.unique(dst.r.values)
    f, u = dst.factor.values, dst.uncertainty.values

    return Correction((z, r),
                      f.reshape(z.size, r.size),
                      u.reshape(z.size, r.size),
                      **kwargs)


def true_photoelect(evt, h5in, true_file):
    this_event_dict = read_mcinfo(h5in, (evt, evt+1))
    event_number    = h5in.root.MC.extents[evt]['evt_number']
    part_dict       = list(this_event_dict.values())[0]

    both      = 0
    interest  = False
    ave_true1 = []
    ave_true2 = []

    for indx, part in part_dict.items():
        if part.name == 'e-' :
            mother = part_dict[part.mother_indx]
            if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                if np.isclose(sum(h.E for h in part.hits), 0.476443, atol=1.e-6):
                    if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                        interest =  True
                        both     += 1

                        if mother.p[1] > 0.:
                            hit_positions = [h.pos for h in part.hits]
                            energies      = [h.E for h in part.hits]
                            energy1       = sum(energies)

                            if energy1 != 0.:
                                ave_true1 = np.average(hit_positions, axis=0, weights=energies)

                        else:
                            hit_positions = [h.pos for h in part.hits]
                            energies      = [h.E for h in part.hits]
                            energy2       = sum(energies)

                            if energy2 != 0.:
                                ave_true2 = np.average(hit_positions, axis=0, weights=energies)

    return both, ave_true1, ave_true2


def true_photoelect_compton(h5in, true_file, evt):

    this_event_dict = read_mcinfo(h5in, (evt, evt+1))
    event_number    = h5in.root.MC.extents[evt]['evt_number']
    part_dict       = list(this_event_dict.values())[0]

    interest1 = False
    interest2 = False
    ave_true1 = []
    ave_true2 = []
    min_time1 = 100000
    min_time2 = 100000

    for indx, part in part_dict.items():
        if part.name == 'e-':
            mother = part_dict[part.mother_indx]
            if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                    if mother.p[1] > 0.:
                        interest1 = True
                        times     = [h.time for h in part.hits]
                        if len(times)==0:
                            continue
                        min_t   = min(times)
                        min_pos = times.index(min_t)
                        if min_t < min_time1:
                            min_time1     = min_t
                            hit_positions = [h.pos for h in part.hits]
                            energies      = [h.E   for h in part.hits]
                            if energies[min_pos] != 0:
                                ave_true1 = hit_positions[min_pos]
                        else:
                            continue

                    else:
                        interest2 = True
                        times     = [h.time for h in part.hits]
                        if len(times)==0:
                            continue
                        min_t   = min(times)
                        min_pos = times.index(min_t)
                        if min_t < min_time2:
                            min_time2     = min_t
                            hit_positions = [h.pos for h in part.hits]
                            energies      = [h.E   for h in part.hits]
                            if energies[min_pos] != 0:
                                ave_true2 = hit_positions[min_pos]
                        else:
                            continue

        elif part.name == 'gamma' and part.primary:
            if part.p[1] > 0:
                times = [h.time for h in part.hits]
                if len(times) == 0:
                    continue
                min_t   = min(times)
                min_pos = times.index(min_t)
                if min_t < min_time1:
                    min_time1     = min_t
                    hit_positions = [h.pos for h in part.hits]
                    energies      = [h.E   for h in part.hits]
                    if energies[min_pos] != 0:
                        ave_true1 = hit_positions[min_pos] #The first interaction of the gamma
            else:
                times = [h.time for h in part.hits]
                if len(times)==0:
                    continue
                min_t   = min(times)
                min_pos = times.index(min_t)
                if min_t < min_time2:
                    min_time2     = min_t
                    hit_positions = [h.pos for h in part.hits]
                    energies      = [h.E   for h in part.hits]
                    if energies[min_pos] != 0:
                        ave_true2 = hit_positions[min_pos]

    return interest1, interest2, ave_true1, ave_true2


def sensor_classification(i1, i2, ave_true1, ave_true2, sens_pos, sens_pos_cyl, sns_over_thr, charges_over_thr):
    """This function calculates the ID of the closest sensor to the true position.
    Then the ring is divided in two sections with the corresponding charge of each one.
    The total charge and the position of each sensor is returned.
    """
    closest1 = closest2 = 0.

    if not i1 or not i2:
        if len(ave_true1) != 0 and len(ave_true2) == 0:
            closest1 = find_closest_sipm(ave_true1[0], ave_true1[1], ave_true1[2],
                                            sens_pos, sns_over_thr, charges_over_thr)

        elif len(ave_true1) == 0 and len(ave_true2) != 0:
            closest2 = find_closest_sipm(ave_true2[0], ave_true2[1], ave_true2[2],
                                            sens_pos, sns_over_thr, charges_over_thr)
    #elif i1 and i2:
    else:
        closest1 = find_closest_sipm(ave_true1[0], ave_true1[1], ave_true1[2],
                                        sens_pos, sns_over_thr, charges_over_thr)
        closest2 = find_closest_sipm(ave_true2[0], ave_true2[1], ave_true2[2],
                                        sens_pos, sns_over_thr, charges_over_thr)

    ampl1    = ampl2  =  0.
    count1   = count2 = 0
    pos1     = []
    pos2     = []
    pos1_cyl = []
    pos2_cyl = []
    q1       = []
    q2       = []

    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos     = sens_pos    [sns_id]
        pos_cyl = sens_pos_cyl[sns_id]
        if closest1:
            pos_closest = sens_pos[closest1]
            scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
            if scalar_prod > 0.:
                pos1    .append(pos)
                pos1_cyl.append(pos_cyl)
                q1      .append(charge)
                ampl1   += charge
                count1  += 1

        if closest2:
            pos_closest = sens_pos[closest2]
            scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
            if scalar_prod > 0.:
                pos2    .append(pos)
                pos2_cyl.append(pos_cyl)
                q2      .append(charge)
                ampl2   += charge
                count2  += 1

    return ampl1, ampl2, count1, count2, pos1, pos2, pos1_cyl, pos2_cyl, q1, q2


def create_list_all_charges(wvf_dict, threshold):

    # This function returns a list of tuples with the id of the sensor that
    # detects some optical photons and the corresponding charge it sees

    list_charges = []
    for s_id, wvf in wvf_dict.items():
        if wvf.charges[0] > threshold:
            wvf_tuple = (s_id, wvf.charges[0])
            list_charges.append(wvf_tuple)
        else:
            continue
    list_charges.sort(key=get_key, reverse=True)

    if len(list_charges) == 0:
        raise SipmZeroCharge

    return list_charges


def get_key(item):
    return item[1]

def rel_dis_xy(n, m, sens_pos):
    # This function calculates the relative distance (for x and y) between two sensors
    subst = np.subtract(sens_pos[n], sens_pos[m])
    return np.linalg.norm(subst[0:2])

def sensor_classif(list_charges, sns_over_thr, charges_over_thr, sens_pos, sens_pos_cyl, n_sipm=5):

    list_dist = []
    pos1      = []
    pos2      = []
    pos_cyl1  = []
    pos_cyl2  = []
    q1        = []
    q2        = []
    ampl1     = ampl2  = 0
    count1    = count2 = 0

    n_of_sipm  = list_charges[0:n_sipm]
    first_sipm = n_of_sipm[0][0]

    for i in n_of_sipm[1:n_sipm]:
        list_dist.append(rel_dis_xy(n_of_sipm[0][0], i[0], sens_pos))

    if any(list_dist) < 50:
        imp_sipm    = first_sipm

        for sns_id, charge in zip(sns_over_thr, charges_over_thr):
            pos_imp = sens_pos    [imp_sipm]
            pos     = sens_pos    [sns_id]
            pos_cyl = sens_pos_cyl[sns_id]
            scalar_prod = sum(a*b for a, b in zip(pos, pos_imp))
            if scalar_prod > 0.:
                pos1    .append(pos)
                pos_cyl1.append(pos_cyl)
                q1      .append(charge)
                ampl1   += charge
                count1  += 1
            else:
                pos2    .append(pos)
                pos_cyl2.append(pos_cyl)
                q2      .append(charge)
                ampl2   += charge
                count2  += 1
        return pos1, pos2, pos_cyl1, pos_cyl2, q1, q2, ampl1, ampl2, count1, count2
    else:
        raise SipmZeroCharge
