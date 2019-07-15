import os
import sys
import math
import tables as tb
import numpy  as np

from   antea.io.mc_io_tb                 import read_mcsns_response
from   invisible_cities.io.mcinfo_io  import read_mcinfo

sys.path.append('/data5/users/carmenromo/PETALO/PETit/PETit-ring/Christoff_sim/compton')
import analysis_utils  as ats


def get_coord_cyl(cart_vect):
    r   = np.sqrt(cart_vect[0]*cart_vect[0] + cart_vect[1]*cart_vect[1])
    phi = np.arctan2(cart_vect[1], cart_vect[0])
    z   = cart_vect[2]
    return np.array([r, phi, z])

def get_coord_cart(cyl_vect):
    x = cyl_vect[0]*np.cos(cyl_vect[1])
    y = cyl_vect[0]*np.sin(cyl_vect[1])
    z = cyl_vect[2]
    return np.array([x, y, z])

def single_sensor_classif(i1, i2, ave_true1, ave_true2, sens_pos, sens_pos_cyl, sns_over_thr, charges_over_thr):
    # i1, i2: only one of them can be True!!
    if i1==i2:
        raise ValueError

    closest    = 0.
    sig_pgamma = False
    ave_true   = []
    if i1 and not i2:
        closest    = ats.find_closest_sipm(ave_true1[0], ave_true1[1], ave_true1[2],
                                           sens_pos, sns_over_thr, charges_over_thr)
        sig_pgamma = True
        ave_true   = ave_true1

    elif i2 and not i1:
        closest    = ats.find_closest_sipm(ave_true2[0], ave_true2[1], ave_true2[2],
                                           sens_pos, sns_over_thr, charges_over_thr)
        sig_pgamma = False
        ave_true   = ave_true2

    ampl1  = 0.
    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos     = sens_pos    [sns_id]
        pos_cyl = sens_pos_cyl[sns_id]
        pos_closest = sens_pos[closest]
        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
        if scalar_prod > 0.:
            ampl1  += charge

    return sig_pgamma, ampl1, ave_true

def get_gamma_p(h5in, evt, sig):
    this_event_dict = read_mcinfo(h5in, (evt, evt+1))
    part_dict       = list(this_event_dict.values())[0]
    p1 = 0
    p2 = 0
    for indx, part in part_dict.items():
        if part.primary:
            if part.p[1]>0: p1 = part.p
            else:           p2 = part.p
    if sig: return p1
    else:   return p2

def get_theta(p1, p2):
    scalar_prod = np.dot(-p1, p2) #I take the opposite of p1!!!!
    mod1        = np.linalg.norm(p1)
    mod2        = np.linalg.norm(p2)
    theta       = np.arccos(scalar_prod/(mod1*mod2))
    return theta

def get_axis(p1, p2):
    vect_prod = np.cross(p1, p2)
    mod       = np.linalg.norm(vect_prod)
    return vect_prod/mod

def rot_matrix(theta, axis): # theta in radians!!
    r11 = math.cos(theta) + (axis[0]*axis[0])*(1 - math.cos(theta))
    r12 = axis[0]*axis[1]*(1 - math.cos(theta)) - axis[2]*math.sin(theta)
    r13 = axis[0]*axis[2]*(1 - math.cos(theta)) + axis[1]*math.sin(theta)
    r21 = axis[1]*axis[0]*(1 - math.cos(theta)) + axis[2]*math.sin(theta)
    r22 = math.cos(theta) + (axis[1]*axis[1])*(1 - math.cos(theta))
    r23 = axis[1]*axis[2]*(1 - math.cos(theta)) - axis[0]*math.sin(theta)
    r31 = axis[2]*axis[0]*(1 - math.cos(theta)) - axis[1]*math.sin(theta)
    r32 = axis[2]*axis[1]*(1 - math.cos(theta)) + axis[0]*math.sin(theta)
    r33 = math.cos(theta) + (axis[2]*axis[2])*(1 - math.cos(theta))
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

def apply_rot(theta, axis, p2):
    return np.dot(rot_matrix(theta, axis), p2)

def single_sensor_classif2(ave_true, sens_pos, sens_pos_cyl, sns_over_thr, charges_over_thr):

    closest = ats.find_closest_sipm(ave_true[0], ave_true[1], ave_true[2],
                                    sens_pos, sns_over_thr, charges_over_thr)

    ampl1    = 0
    count1   = 0
    pos1     = []
    pos1_cyl = []
    q1       = []

    for sns_id, charge in zip(sns_over_thr, charges_over_thr):
        pos     = sens_pos    [sns_id]
        pos_cyl = sens_pos_cyl[sns_id]
        pos_closest = sens_pos[closest]
        scalar_prod = sum(a*b for a, b in zip(pos, pos_closest))
        if scalar_prod > 0.:
            ## I will store the position in both coord!! cartesians and cylindrical!!!
            pos1    .append(pos)
            pos1_cyl.append(pos_cyl)
            q1      .append(charge)
            ampl1   += charge
            count1  += 1

    return ampl1, count1, pos1, pos1_cyl, q1

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

    positions     = []
    qs            = []

    for th in list_thrs:
        indices_over_thr = sns_q > th
        pos_over_thr     = sns_pos[indices_over_thr]
        charges_over_thr = sns_q[indices_over_thr]

        if len(charges_over_thr) == 0:
            return [], []

        positions.append(np.array(pos_over_thr))
        qs       .append(np.array(charges_over_thr))

    return positions, qs
