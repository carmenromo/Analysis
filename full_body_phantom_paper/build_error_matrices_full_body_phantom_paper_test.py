import os
import sys
import argparse
import numpy as np

import antea.database.load_db as db


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path' ,   help = "input files path"    )
    parser.add_argument('out_path',   help = "output matrices path")
    parser.add_argument('identifier', help = "Identifier of the conf")
    return parser.parse_args()

arguments  = parse_args(sys.argv)
filepath   = arguments.in_path
folder     = arguments.out_path
identifier = arguments.identifier

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

sns_response1, sns_response2    = [], []

first_sipm1, first_sipm2 = [], []
first_time1, first_time2 = [], []

true_t1, true_t2                     = [], []
touched_sipms1, touched_sipms2       = [], []
phot1, phot2                         = [], []
max_hit_distance1, max_hit_distance2 = [], []
hit_energy1, hit_energy2             = [], []

event_ids = []

for filename in os.listdir(filepath):
    if filename.endswith('.npz'):
        print(filename)
        my_file = filepath+filename
        d       = np.load(my_file)

        true_r1   = np.concatenate((true_r1,   d['a_true_r1']))
        true_phi1 = np.concatenate((true_phi1, d['a_true_phi1']))
        true_z1   = np.concatenate((true_z1,   d['a_true_z1']))

        reco_r1   = np.concatenate((reco_r1,   d['a_reco_r1']))
        reco_phi1 = np.concatenate((reco_phi1, d['a_reco_phi1']))
        reco_z1   = np.concatenate((reco_z1,   d['a_reco_z1']))

        true_r2   = np.concatenate((true_r2,   d['a_true_r2']))
        true_phi2 = np.concatenate((true_phi2, d['a_true_phi2']))
        true_z2   = np.concatenate((true_z2,   d['a_true_z2']))

        reco_r2   = np.concatenate((reco_r2,   d['a_reco_r2']))
        reco_phi2 = np.concatenate((reco_phi2, d['a_reco_phi2']))
        reco_z2   = np.concatenate((reco_z2,   d['a_reco_z2']))

        true_t1 = np.concatenate((true_t1, d['a_true_time1']))
        true_t2 = np.concatenate((true_t2, d['a_true_time2']))

        touched_sipms1 = np.concatenate((touched_sipms1, d['a_touched_sipms1']))
        touched_sipms2 = np.concatenate((touched_sipms2, d['a_touched_sipms2']))
        sns_response1  = np.concatenate((sns_response1,  d['a_sns_response1']))
        sns_response2  = np.concatenate((sns_response2,  d['a_sns_response2']))

        phot1             = np.concatenate((phot1, d['a_photo1']))
        phot2             = np.concatenate((phot2, d['a_photo2']))
        max_hit_distance1 = np.concatenate((max_hit_distance1, d['a_max_hit_distance1']))
        max_hit_distance2 = np.concatenate((max_hit_distance2, d['a_max_hit_distance2']))
        hit_energy1       = np.concatenate((hit_energy1, d['a_hit_energy1']))
        hit_energy2       = np.concatenate((hit_energy2, d['a_hit_energy2']))

        event_ids = np.concatenate((event_ids, d['a_event_ids']))

        first_time1 = np.concatenate((first_time1, d[f'a_first_time1_1']))
        first_time2 = np.concatenate((first_time2, d[f'a_first_time2_1']))

        first_sipm1 = np.concatenate((first_sipm1, d[f'a_first_sipm1_1']))
        first_sipm2 = np.concatenate((first_sipm2, d[f'a_first_sipm2_1']))


true_r1   = np.array(true_r1)
true_phi1 = np.array(true_phi1)
true_z1   = np.array(true_z1)

reco_r1   = np.array(reco_r1)
reco_phi1 = np.array(reco_phi1)
reco_z1   = np.array(reco_z1)

true_r2   = np.array(true_r2)
true_phi2 = np.array(true_phi2)
true_z2   = np.array(true_z2)

reco_r2   = np.array(reco_r2)
reco_phi2 = np.array(reco_phi2)
reco_z2   = np.array(reco_z2)

true_t1 = np.array(true_t1)
true_t2 = np.array(true_t2)

first_time1 = np.array(first_time1)
first_time2 = np.array(first_time2)
first_sipm1 = np.array(first_sipm1)
first_sipm2 = np.array(first_sipm2)

sns_response1 = np.array(sns_response1)
sns_response2 = np.array(sns_response2)

phot1             = np.array(phot1)
phot2             = np.array(phot2)
max_hit_distance1 = np.array(max_hit_distance1)
max_hit_distance2 = np.array(max_hit_distance2)
hit_energy1       = np.array(hit_energy1)
hit_energy2       = np.array(hit_energy2)

event_ids = np.array(event_ids)

true_x1 = true_r1 * np.cos(true_phi1)
reco_x1 = reco_r1 * np.cos(reco_phi1)
true_y1 = true_r1 * np.sin(true_phi1)
reco_y1 = reco_r1 * np.sin(reco_phi1)
true_x2 = true_r2 * np.cos(true_phi2)
reco_x2 = reco_r2 * np.cos(reco_phi2)
true_y2 = true_r2 * np.sin(true_phi2)
reco_y2 = reco_r2 * np.sin(reco_phi2)


### change by hand phi reconstructed as true=~3.14, reco~=-3.14
reco_phi1[np.abs(reco_phi1 - true_phi1) > 6.] = -reco_phi1[np.abs(reco_phi1 - true_phi1) > 6.]
reco_phi2[np.abs(reco_phi2 - true_phi2) > 6.] = -reco_phi2[np.abs(reco_phi2 - true_phi2) > 6.]

true_x   = np.concatenate((true_x1, true_x2))
true_y   = np.concatenate((true_y1, true_y2))
true_r   = np.concatenate((true_r1, true_r2))
true_phi = np.concatenate((true_phi1, true_phi2))
true_z   = np.concatenate((true_z1, true_z2))

reco_x   = np.concatenate((reco_x1, reco_x2))
reco_y   = np.concatenate((reco_y1, reco_y2))
reco_r   = np.concatenate((reco_r1, reco_r2))
reco_phi = np.concatenate((reco_phi1, reco_phi2))
reco_z   = np.concatenate((reco_z1, reco_z2))

true_t = np.concatenate((true_t1, true_t2))

sns_response     = np.concatenate((sns_response1, sns_response2))
phot             = np.concatenate((phot1, phot2))
max_hit_distance = np.concatenate((max_hit_distance1, max_hit_distance2))
hit_energy       = np.concatenate((hit_energy1, hit_energy2))

n_int = len(true_x) # number of interactions



def diff_and_concatenate(true1, true2, reco1, reco2):
    d1 = true1 - reco1
    d2 = true2 - reco2
    diff_matrix = np.concatenate((d1, d2))
    return diff_matrix

diff_r_matrix   = diff_and_concatenate(true_r1,   true_r2,   reco_r1,   reco_r2)
diff_phi_matrix = diff_and_concatenate(true_phi1, true_phi2, reco_phi1, reco_phi2)
diff_z_matrix   = diff_and_concatenate(true_z1,   true_z2,   reco_z1,   reco_z2)


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

speed_in_vacuum = 0.299792458# * units.mm / units.ps
ave_speed_in_LXe = 0.210 #* units.mm / units.ps


### Positions
pos_1 = np.array([reco_x1, reco_y1, reco_z1]).transpose()
pos_2 = np.array([reco_x2, reco_y2, reco_z2]).transpose()
dist1 = []
dist2 = []

sipm_pos_1 = np.array([DataSiPM_idx.loc[first_sipm1].X, DataSiPM_idx.loc[first_sipm1].Y, DataSiPM_idx.loc[first_sipm1].Z]).transpose()
sipm_pos_2 = np.array([DataSiPM_idx.loc[first_sipm2].X, DataSiPM_idx.loc[first_sipm2].Y, DataSiPM_idx.loc[first_sipm2].Z]).transpose()

### Distance of the interaction point from the SiPM seeing the first photon
dist1.append(np.linalg.norm(np.subtract(pos_1, sipm_pos_1), axis=1))
dist2.append(np.linalg.norm(np.subtract(pos_2, sipm_pos_2), axis=1))

reco_t1   = np.array(first_time1) - (np.array(dist1)/ave_speed_in_LXe)
reco_t2   = np.array(first_time2) - (np.array(dist2)/ave_speed_in_LXe)

diff_t_matrix      = []
diff_reco_t_matrix = []
diff_t_matrix     .append((diff_and_concatenate(true_t1, true_t2, first_time1, first_time2)).flatten())
diff_reco_t_matrix.append((diff_and_concatenate(true_t1, true_t2,     reco_t1,     reco_t2)).flatten())


d1 = 1 #mm Phot like events
sel_phot_like  = max_hit_distance < d1
sel_compt_like = max_hit_distance > d1

print(f'Number of interactions for phot = {len(true_x[sel_phot_like])}')


def get_bins(coord_range, err_range_phot, err_range_compt, coord_width, err_width_phot, err_width_compt):
    coord_bins      = int((coord_range    [1] - coord_range    [0])/coord_width)
    err_bins_phot   = int((err_range_phot [1] - err_range_phot [0])/err_width_phot)
    err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
    return coord_bins, err_bins_phot, err_bins_compt


precision = .3 # mm
r_max     = 409.7

r_range           = (380, 409.7)
r_err_range_phot  = (-5, 10)
r_err_range_compt = (-30, 30)
r_width           = precision
r_err_width_phot  = precision
r_err_width_compt = precision
r_bins, r_err_bins_phot, r_err_bins_compt = get_bins(r_range,
                                                     r_err_range_phot,
                                                     r_err_range_compt,
                                                     r_width,
                                                     r_err_width_phot,
                                                     r_err_width_compt)

phi_range           = (-3.15, 3.15)
phi_err_range_phot  = (-0.01, 0.01)
phi_err_range_compt = (-0.15, 0.15)
phi_width           = precision/r_max
phi_err_width_phot  = precision/r_max
phi_err_width_compt = precision/r_max
phi_bins, phi_err_bins_phot, phi_err_bins_compt = get_bins(phi_range,
                                                           phi_err_range_phot,
                                                           phi_err_range_compt,
                                                           phi_width,
                                                           phi_err_width_phot,
                                                           phi_err_width_compt)

z_range           = (-975, 975)
z_err_range_phot  = (-3, 3)
z_err_range_compt = (-100, 100)
z_width           = precision
z_err_width_phot  = precision
z_err_width_compt = precision
z_bins, z_err_bins_phot, z_err_bins_compt = get_bins(z_range,
                                                     z_err_range_phot,
                                                     z_err_range_compt,
                                                     z_width,
                                                     z_err_width_phot,
                                                     z_err_width_compt)

t_precision       = 5 # ps
t_range           = (1000, 3500)
t_err_range_phot  = (-400, 50)
t_err_range_compt = (-480, 300)
t_width           = t_precision
t_err_width_phot  = t_precision
t_err_width_compt = t_precision
t_bins, t_err_bins_phot, t_err_bins_compt = get_bins(t_range,
                                                     t_err_range_phot,
                                                     t_err_range_compt,
                                                     t_width,
                                                     t_err_width_phot,
                                                     t_err_width_compt)

print('')
print('**** R ****')
print(f'Number bins: true r = {r_bins}, err phot = {r_err_bins_phot}, err compt = {r_err_bins_compt}')
print('')
print('**** Phi ****')
print(f'Number bins: true phi = {phi_bins}, true r = {r_bins}, err phot = {phi_err_bins_phot}, err compt = {phi_err_bins_compt}')
print('')
print('**** Z ****')
print(f'Number bins: true z = {z_bins}, true r = {r_bins}, err phot = {z_err_bins_phot}, err compt = {z_err_bins_compt}')
pint('')
print('**** T ****')
print(f'Number bins: true t = {t_bins}, err phot = {t_err_bins_phot}, err compt = {t_err_bins_compt}')


def extract_info_from_edges_2d(edges):
    eff    = np.array([1])
    xedges = edges[0]
    yedges = edges[1]
    xmin   = xedges[0]; xmin = np.array(xmin)
    ymin   = yedges[0]; ymin = np.array(ymin)
    dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
    dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
    return eff, xedges, yedges, xmin, ymin, dx, dy

def extract_info_from_edges_3d(edges):
    eff    = np.array([1])
    xedges = edges[0]
    yedges = edges[1]
    zedges = edges[2]
    xmin   = xedges[0]; xmin = np.array(xmin)
    ymin   = yedges[0]; ymin = np.array(ymin)
    zmin   = zedges[0]; zmin = np.array(zmin)
    dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
    dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
    dz     = zedges[1:]-zedges[:-1]; dz = np.array(dz[0])
    return eff, xedges, yedges, zedges, xmin, ymin, zmin, dx, dy, dz


### R
## photoelectric-like events
h_r, edges_r = np.histogramdd((true_r       [sel_phot_like],
                               diff_r_matrix[sel_phot_like]),
                               bins  = (r_bins,  r_err_bins_phot),
                               range = (r_range, r_err_range_phot))


eff, xedges, yedges, xmin, ymin, dx, dy = extract_info_from_edges_2d(edges_r)
file_name = folder + f'errmat_{identifier}_r_phot_like.npz'
np.savez(file_name, errmat=h_r, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Phot-like')
a = np.sum(h_r, axis=1)
print(a.shape, np.count_nonzero(a))

## compton-like events
h_r, edges_r = np.histogramdd((true_r       [sel_compt_like],
                               diff_r_matrix[sel_compt_like]),
                               bins  = (r_bins,  r_err_bins_compt),
                               range = (r_range, r_err_range_compt))
eff, xedges, yedges, xmin, ymin, dx, dy = extract_info_from_edges_2d(edges_r)
file_name = folder + f'errmat_{identifier}_r_compt_like.npz'
np.savez(file_name, errmat=h_r, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)


print('Compt-like')
a = np.sum(h_r, axis=1)
print(a.shape, np.count_nonzero(a))


### Phi
## photoelectric-like events
h_phi, edges_phi = np.histogramdd((true_phi       [sel_phot_like],
                                   true_r         [sel_phot_like],
                                   diff_phi_matrix[sel_phot_like]),
                                   bins  = (phi_bins,  r_bins,  phi_err_bins_phot),
                                   range = (phi_range, r_range, phi_err_range_phot))

eff, xedges, yedges, zedges, xmin, ymin, zmin, dx, dy, dz = extract_info_from_edges_3d(edges_phi)
file_name = folder + f'errmat_{identifier}_phi_phot_like.npz'
np.savez(file_name, errmat=h_phi, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Phot-like')
a = np.sum(h_phi, axis=2)
print(a.shape, a.shape[0]*a.shape[1], np.count_nonzero(a))



## compton-like events
h_phi, edges_phi = np.histogramdd((true_phi       [sel_compt_like],
                                   true_r         [sel_compt_like],
                                   diff_phi_matrix[sel_compt_like]),
                                   bins  = (phi_bins,  r_bins,  phi_err_bins_compt),
                                   range = (phi_range, r_range, phi_err_range_compt))

eff, xedges, yedges, zedges, xmin, ymin, zmin, dx, dy, dz = extract_info_from_edges_3d(edges_phi)
file_name = folder + f'errmat_{identifier}_phi_compt_like.npz'
np.savez(file_name, errmat=h_phi, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Compt-like')
a = np.sum(h_phi, axis=2)
print(a.shape, a.shape[0]*a.shape[1], np.count_nonzero(a))



### Z
## photoelectric-like events
h_z, edges_z = np.histogramdd((true_z       [sel_phot_like],
                               true_r       [sel_phot_like],
                               diff_z_matrix[sel_phot_like]),
                               bins  = (z_bins,  r_bins,  z_err_bins_phot),
                               range = (z_range, r_range, z_err_range_phot))

eff, xedges, yedges, zedges, xmin, ymin, zmin, dx, dy, dz = extract_info_from_edges_3d(edges_z)
file_name = folder + f'errmat_{identifier}_z_phot_like.npz'
np.savez(file_name, errmat=h_z, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Phot-like')
a = np.sum(h_z, axis=2)
print(a.shape, a.shape[0]*a.shape[1], np.count_nonzero(a))

## compton-like events
h_z, edges_z = np.histogramdd((true_z       [sel_compt_like],
                               true_r       [sel_compt_like],
                               diff_z_matrix[sel_compt_like]),
                               bins  = (z_bins,  r_bins,  z_err_bins_compt),
                               range = (z_range, r_range, z_err_range_compt))

eff, xedges, yedges, zedges, xmin, ymin, zmin, dx, dy, dz = extract_info_from_edges_3d(edges_z)
file_name = folder + f'errmat_{identifier}_z_compt_like.npz'
np.savez(file_name, errmat=h_z, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Compt-like')
a = np.sum(h_z, axis=2)
print(a.shape, a.shape[0]*a.shape[1], np.count_nonzero(a))



### T
## photoelectric-like events
h_t, edges_t = np.histogramdd((true_t               [sel_phot_like],
                               diff_reco_t_matrix[0][sel_phot_like]),
                               bins  = (t_bins,  t_err_bins_phot),
                               range = (t_range, t_err_range_phot))

eff, xedges, yedges, xmin, ymin, dx, dy = extract_info_from_edges_2d(edges_t)
file_name = folder + f'errmat_{identifier}_t_thr0pes_phot_like.npz'
np.savez(file_name, errmat=h_t, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Phot-like')
a = np.sum(h_t, axis=1)
print(a.shape, np.count_nonzero(a))

## compton-like events
h_t, edges_t = np.histogramdd((true_t               [sel_compt_like],
                               diff_reco_t_matrix[0][sel_compt_like]),
                               bins  = (t_bins,  t_err_bins_compt),
                               range = (t_range, t_err_range_compt))

eff, xedges, yedges, xmin, ymin, dx, dy = extract_info_from_edges_2d(edges_t)
file_name = folder + f'errmat_{identifier}_t_thr0pes_compt_like.npz'
np.savez(file_name, errmat=h_t, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Compt-like')
a = np.sum(h_t, axis=1)
print(a.shape, np.count_nonzero(a))
