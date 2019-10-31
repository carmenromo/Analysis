import tables as tb
import numpy  as np

from invisible_cities.io  .mcinfo_io  import read_mcinfo
from invisible_cities.core.exceptions import NoHits


def true_photoelect(h5in, true_file, evt, compton=False, energy=False):
    """Returns the position of the true photoelectric energy deposition
    calculated with barycenter algorithm.
    It allows the possibility of including compton events and
    returning the energy.
    """
    this_event_dict = read_mcinfo(h5in, (evt, evt+1))
    part_dict       = list(this_event_dict.values())[0]

    ave_true1, ave_true2 = [], []
    energy1  , energy2   =  0,  0

    for indx, part in part_dict.items():
        if part.name == 'e-' :
            mother = part_dict[part.mother_indx]
            if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                if mother.primary and np.isclose(mother.E*1000., 510.999, atol=1.e-3):
                    if compton==True: pass
                    else:
                        if sum(h.E for h in part.hits) >= 0.476443: pass
                        else: continue

                    if mother.p[1] > 0.: ave_true1, energy1 = get_true_pos(part)
                    else:                ave_true2, energy2 = get_true_pos(part)
    if energy: return ave_true1, ave_true2, energy1, energy2
    else:      return ave_true1, ave_true2
    

def get_true_pos(part):
    hit_positions = [h.pos for h in part.hits]
    energies      = [h.E   for h in part.hits]
    energy        = sum(energies)
    if energy: return np.average(hit_positions, axis=0, weights=energies), energy
    else: raise NoHits


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


def sensor_position(h5in):
    """A dictionary that stores the position of all the sensors
    in cartesian coordinates is created.
    """
    sipms    = h5in.root.MC.sensor_positions[:]
    sens_pos = {}
    for sipm in sipms:
        sens_pos[sipm[0]] = (sipm[1], sipm[2], sipm[3])
    return sens_pos


def sensor_position_cyl(h5in):
    """A dictionary that stores the position of all the sensors
    in cylindrical coordinates is created.
    """
    sipms        = h5in.root.MC.sensor_positions[:]
    sens_pos_cyl = {}
    for sipm in sipms:
        sens_pos_cyl[sipm[0]] = (np.sqrt(sipm[1]*sipm[1] + sipm[2]*sipm[2]),
                                 np.arctan2(sipm[2], sipm[1]),
                                 sipm[3])
    return sens_pos_cyl


def sensors_info(ave_true, sens_pos, sens_pos_cyl, sns_over_thr, charges_over_thr):

    """For a given true position of an event, returns all the information of the sensors
    for the corresponding half of the ring.

    Parameters
    ----------
    ave_true         : np.array
    Position of the true hits (cart coordinates).
    sens_pos         : dict
    Contains the position of each sensor (cart coordinates).
    sens_pos_cyl     : dict
    Contains the position of each sensor (cyl coordinates).
    sns_over_thr     : np.array
    IDs of the sensors that detected charge above a certain threshold.
    charges_over_thr : np.array
    Charges of the sensors above a certain threshold

    Returns
    -------
    ampl1    : int
    Total charge detected for this single event.
    count1   : int
    Number of sensors that detected charge.
    pos1     : np.array
    Position of every sensor that detected some charge (cart coordinates).
    pos1_cyl : np.array
    Position of every sensor that detected some charge (cyl coordinates).
    q1       : np.array
    Charge detected by every sensor.
    """

    ampl1    = 0
    count1   = 0
    pos1     = []
    pos1_cyl = []
    q1       = []

    if not len(ave_true):
        return ampl1, count1, pos1, pos1_cyl, q1

    closest = find_closest_sipm(ave_true[0], ave_true[1], ave_true[2],
                                sens_pos, sns_over_thr, charges_over_thr)

    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos         = sens_pos    [sns_id]
        pos_cyl     = sens_pos_cyl[sns_id]
        pos_closest = sens_pos    [closest]
        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
        if scalar_prod > 0.:
            pos1    .append(pos)
            pos1_cyl.append(pos_cyl)
            q1      .append(charge)
            ampl1   += charge
            count1  += 1

    return ampl1, count1, pos1, pos1_cyl, q1


def find_SiPMs_over_threshold(this_event_wvf, threshold):

    sns_dict = list(this_event_wvf.values())[0]
    tot_charges = np.array(list(map(lambda x: sum(x.charges), list(sns_dict.values()))))
    sns_ids = np.array(list(sns_dict.keys()))

    indices_over_thr = (tot_charges > threshold)
    sns_over_thr = sns_ids[indices_over_thr]
    charges_over_thr = tot_charges[indices_over_thr]

    return sns_over_thr, charges_over_thr


def get_r_and_var_phi(ave_true, cyl_pos, q):
    r         = np.sqrt(ave_true[0]**2 + ave_true[1]**2)
    phi_pos   = np.array([el[1] for el in cyl_pos])
    diff_sign = min(phi_pos) < 0 < max(phi_pos)
    if diff_sign & (np.abs(np.min(phi_pos))>np.pi/2):
        phi_pos[phi_pos<0] = np.pi + np.pi + phi_pos[phi_pos<0]
    mean_phi = np.average(phi_pos, weights=q)
    var_phi  = np.average((phi_pos - mean_phi)**2, weights=q)
    return r, var_phi


def get_r_and_var_phi2(ave_true, cyl_pos, q):
    r        = np.sqrt(ave_true[0]**2 + ave_true[1]**2)
    phi_pos  = np.array([el[1] for el in cyl_pos])
    mean_phi = np.average(phi_pos, weights=q)
    var_phi  = np.average((phi_pos - mean_phi)**2, weights=q)
    return r, var_phi


def get_var_phi(phi_pos, q):
    diff_sign = min(phi_pos) < 0 < max(phi_pos)
    if diff_sign & (np.abs(np.min(phi_pos))>np.pi/2):
        phi_pos[phi_pos<0] = np.pi + np.pi + phi_pos[phi_pos<0]
    mean_phi = np.average(phi_pos, weights=q)
    var_phi  = np.average((phi_pos - mean_phi)**2, weights=q)
    return var_phi


def sensor_classification(ave_true1, ave_true2, sens_pos, sens_pos_cyl, sns_over_thr, charges_over_thr):
    """This function calculates the ID of the closest sensor to the true position.
    Then the ring is divided in two sections with the corresponding charge of each one.
    The total charge and the position of each sensor is returned.
    """
    closest1 = closest2 = 0.

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

    ampl1   = ampl2    = 0.
    pos1    , pos2     = [], []
    pos1_cyl, pos2_cyl = [], []
    q1      , q2       = [], []

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

        if closest2:
            pos_closest = sens_pos[closest2]
            scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
            if scalar_prod > 0.:
                pos2    .append(pos)
                pos2_cyl.append(pos_cyl)
                q2      .append(charge)
                ampl2   += charge

    return ampl1, ampl2, pos1, pos2, pos1_cyl, pos2_cyl, q1, q2


def charges_pass_thr(h5in, true_file, evt, ave_true, th_r, th_phi, th_z, th_e, sens_pos, sens_pos_cyl):
    this_event_dict = read_mcinfo(h5in, (evt, evt+1))
    this_event_wvf  = read_mcsns_response(true_file, (evt, evt+1))
    part_dict       = list(this_event_dict.values())[0]

    sns_dict    = list(this_event_wvf.values())[0]
    tot_charges = np.array(list(map(lambda x: sum(x.charges), sns_dict.values())))
    sns_ids     = np.array(list(sns_dict.keys()))

    list_thrs = [th_r, th_phi, th_z, th_e]

    ampls    = []
    counts   = []
    poss     = []
    poss_cyl = []
    qs       = []

    for i, th in enumerate(list_thrs):
        indices_over_thr = (tot_charges > th)
        sns_over_thr     = sns_ids    [indices_over_thr]
        charges_over_thr = tot_charges[indices_over_thr]

        if len(charges_over_thr) == 0:
            continue

        ampl, count, pos, pos_cyl, q = single_sensor_classif2(ave_true,
                                                              sens_pos,
                                                              sens_pos_cyl,
                                                              sns_over_thr,
                                                              charges_over_thr)
        ampls   .append(ampl )
        counts  .append(count)
        poss    .append(np.array(pos))
        poss_cyl.append(np.array(pos_cyl))
        qs      .append(np.array(q))

    if len(ampls)!=len(list_thrs) or len(counts)!=len(list_thrs) or len(poss)!=len(list_thrs) or len(poss_cyl)!=len(list_thrs) or len(qs)!=len(list_thrs):
        raise ValueError
    return ampls, counts, poss, poss_cyl, qs


def reco_pos_single(true_pos, sns_q, sns_pos, th_r, th_phi, th_z):
    list_thrs = [th_r, th_phi, th_z]
    positions = []
    qs        = []
    for th in list_thrs:
        indices_over_thr = sns_q > th
        pos_over_thr     = sns_pos[indices_over_thr]
        charges_over_thr = sns_q  [indices_over_thr]
        if len(charges_over_thr) == 0:
            return [], []

        positions.append(np.array(pos_over_thr))
        qs       .append(np.array(charges_over_thr))
    
    return positions, qs


def select_true_pos_from_charge(sns_over_thr, charges_over_thr, charge_range, sens_pos, part_dict):
    """
    This functions returns a lot of things:
    interest1, interest2: boolean, it's true if the event of the emisphere 1/2 is of interest
    pos_true1, pos_true2: the true position of the first interaction of the gamma in the hemisphere 1/2
    gamma1, gamma2: boolean, it's true if the primary gamma originating the interaction in hemisphere 1/2 has py>0
    charges1, charges2: the list of the charges detected by the SiPMs of emisphere 1/2
    positions1, positions2: the list of the positions of the SiPMs of emisphere 1/2
    """

    positions1, positions2 = [], []
    charges1  , charges2   = [], []
    ids1      , ids2       = [], []   

    ### Find the SiPM with maximum charge. The set if sensors around it are labelled as 1
    ### The sensors on the opposite emisphere are labelled as 2.
    max_sns = sns_over_thr[np.argmax(charges_over_thr)]
    max_pos = sens_pos[max_sns]
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos = sens_pos[sns_id]
        scalar_prod = sum(a*b for a, b in zip(pos, max_pos))
        if scalar_prod > 0.:
            charges1  .append(charge)
            positions1.append(pos)
            ids1      .append(sns_id)
        else:
            charges2  .append(charge)
            positions2.append(pos)
            ids2      .append(sns_id)
    q1 = sum(charges1)
    q2 = sum(charges2)


    sel1 = (q1 > charge_range[0]) & (q1 < charge_range[1])
    sel2 = (q2 > charge_range[0]) & (q2 < charge_range[1])
    if not sel1 and not sel2:
        return False, False, [], [], None, None, [], [], [], [], [], []


    ## find the first interactions of the primary gamma(s)
    tvertex_pos = tvertex_neg = -1
    min_pos_pos, min_pos_neg = None, None

    for _, part in part_dict.items():
        if part.name == 'e-':
            if part.initial_volume == 'ACTIVE' and part.final_volume == 'ACTIVE':
                mother = part_dict[part.mother_indx]
                if np.isclose(mother.E*1000., 510.999, atol=1.e-3) and mother.primary:
                    if mother.p[1] > 0.:
                        if tvertex_pos < 0 or part.initial_vertex[3] < tvertex_pos:
                            min_pos_pos = part.initial_vertex[0:3]
                            tvertex_pos = part.initial_vertex[3]
                    else:
                        if tvertex_neg < 0 or part.initial_vertex[3] < tvertex_neg:
                            min_pos_neg = part.initial_vertex[0:3]
                            tvertex_neg = part.initial_vertex[3]


        elif part.name == 'gamma' and part.primary:
            if len(part.hits) > 0:
                if part.p[1] > 0.:
                    times = [h.time for h in part.hits]
                    hit_positions = [h.pos for h in part.hits]
                    min_time = min(times)
                    if min_time < tvertex_pos:
                        min_pos_pos = hit_positions[times.index(min_time)]
                        tvertex_pos = min_time
                else:
                    times = [h.time for h in part.hits]
                    hit_positions = [h.pos for h in part.hits]
                    min_time = min(times)
                    if min_time < tvertex_neg:
                        min_pos_neg = hit_positions[times.index(min_time)]
                        tvertex_neg = min_time

    interest1, interest2 = False, False
    gamma_1, gamma_2 = None, None
    pos_true1, pos_true2 = [], []

    if sel1 and sel2:
        if min_pos_pos is not None and min_pos_neg is not None:
            interest1, interest2 = True, True
            scalar_prod = sum(a*b for a, b in zip(min_pos_pos, max_pos))
            if scalar_prod > 0:
                pos_true1 = min_pos_pos
                pos_true2 = min_pos_neg
                gamma_1   = True
                gamma_2   = False
            else:
                pos_true1 = min_pos_neg
                pos_true2 = min_pos_pos
                gamma_1   = False
                gamma_2   = True

        else:
            print("Houston, we've got a problem 0")

    elif sel1:
        if min_pos_pos is not None:
            scalar_prod = sum(a*b for a, b in zip(min_pos_pos, max_pos))
            if scalar_prod > 0:
                interest1 = True
                pos_true1 = min_pos_pos
                gamma_1   = True

        if min_pos_neg is not None:
            scalar_prod = sum(a*b for a, b in zip(min_pos_neg, max_pos))
            if scalar_prod > 0:
                if interest1:
                    print("Houston, we've got a problem 1: both gammas interact in the same emisphere. This event cannot be used to join singles.")
                    interest1 = False
                    pos_true1 = []
                    gamma_1 = None
                else:
                    interest1 = True
                    pos_true1 = min_pos_neg
                    gamma_1 = False
        if min_pos_pos is None and min_pos_neg is None:
            print("Houston, we've got a problem 2")

    elif sel2:
        if min_pos_pos is not None:
            scalar_prod = sum(a*b for a, b in zip(min_pos_pos, max_pos))
            if scalar_prod <= 0:
                interest2 = True
                pos_true2 = min_pos_pos
                gamma_2 = True

        if min_pos_neg is not None:
            scalar_prod = sum(a*b for a, b in zip(min_pos_neg, max_pos))
            if scalar_prod <= 0:
                if interest2:
                    print("Houston, we've got a problem 3: both gammas interact in the same emisphere. This event cannot be used to join singles.")
                    interest2 = False
                    pos_true2 = []
                    gamma_2 = None
                else:
                    interest2 = True
                    pos_true2 = min_pos_neg
                    gamma_2 = False
        if min_pos_pos is None and min_pos_neg is None:
            print("Houston, we've got a problem 4")


    return interest1, interest2, pos_true1, pos_true2, gamma_1, gamma_2, ids1, ids2, charges1, charges2, positions1, positions2


def find_first_time_of_sensors(sns_dict_tof, ids):
    sns_ids_tof         = np.array(list(sns_dict_tof.keys()))
    sel_sns             = np.isin(-sns_ids_tof, ids)
    tof_ids             = sns_ids_tof[sel_sns]
    tot_charges_tof     = np.array(list(map(lambda x: sum(x.charges), sns_dict_tof.values())))[sel_sns]
    first_timestamp_tof = np.array(list(map(lambda x:     x.times[0], sns_dict_tof.values())))[sel_sns]
    min_t               = min(first_timestamp_tof)
    min_id              = tof_ids[np.where(first_timestamp_tof==min_t)[0][0]]
    return min_t, min_id


def intersect_points_line_and_circ(p1_line, p2_line, r_circ):
    x1, y1, z1 = p1_line
    x2, y2, z2 = p2_line
    m = (y2 - y1)/(x2 - x1)
    xa = (x1*m*m - m*y1 - np.sqrt(r_circ*r_circ*(m*m+1) - x1*x1*m*m - y1*y1 + 2*x1*y1*m)) / (m*m+1)
    xb = (x1*m*m - m*y1 + np.sqrt(r_circ*r_circ*(m*m+1) - x1*x1*m*m - y1*y1 + 2*x1*y1*m)) / (m*m+1)
    
    ya = m * (xa-x1) + y1
    yb = m * (xb-x1) + y1
    
    m2 = (z2-z1)/(x2-x1)
    za = m2*(-r_circ-x1) + z1
    zb = m2*( r_circ-x2) + z1
    
    d1 = np.linalg.norm(np.subtract(p1_line, np.array([xa, ya, za])))
    d2 = np.linalg.norm(np.subtract(p1_line, np.array([xb, yb, zb])))
    if d1 < d2:
        return np.array([xa, ya, za]), np.array([xb, yb, zb])
    else:
        return np.array([xb, yb, zb]), np.array([xa, ya, za])
