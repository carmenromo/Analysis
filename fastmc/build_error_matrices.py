import os
import sys
import numpy as np

import antea.database.load_db as db


filepath = '/Users/carmenromoluque/nexus_petit_analysis/full-body-PET/nest/data_extract_coinc_phantom/'
#filepath = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/fastmc/data_coincidences_phot_info_dist_tof_dif_thr_charge/'

folder = '/Users/carmenromoluque/nexus_petit_analysis/full-body-PET/nest/error_matrices/'
#folder = '/Users/carmenromoluque/nexus_petit_analysis/PETit-ring/fastmc/'


true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []
events = []
sns_response1, sns_response2 = [], []
true_t1, true_t2 = [], []
sipm_t1, sipm_t2         = [[] for i in range(4)], [[] for i in range(4)]
first_sipm1, first_sipm2 = [[] for i in range(4)], [[] for i in range(4)]
phot1, phot2, phot_like1, phot_like2 = [], [], [], []
true_dep_e1, true_dep_e2 = [], []


list_coinc = []
for filename in os.listdir(filepath):
    if filename.endswith('.npz'):
        print(filename)
        my_file = filepath+filename
        d   = np.load(my_file)

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

        sns_response1 = np.concatenate((sns_response1, d['a_sns_response1']))
        sns_response2 = np.concatenate((sns_response2, d['a_sns_response2']))

        phot1       = np.concatenate((phot1, d['a_photo1']))
        phot2       = np.concatenate((phot2, d['a_photo2']))
        phot_like1  = np.concatenate((phot_like1, d['a_photo_like1']))
        phot_like2  = np.concatenate((phot_like2, d['a_photo_like2']))
        true_dep_e1 = np.concatenate((true_dep_e1, d['a_hit_energy1']))
        true_dep_e2 = np.concatenate((true_dep_e2, d['a_hit_energy2']))

        events = np.concatenate((events, d['a_event_ids']))

        for i in range(4):
            sipm_t1[i] = np.concatenate((sipm_t1[i], d[f'a_first_time1_thr{i}']))
            sipm_t2[i] = np.concatenate((sipm_t2[i], d[f'a_first_time2_thr{i}']))

            first_sipm1[i] = np.concatenate((first_sipm1[i], d[f'a_first_sipm1_thr{i}']))
            first_sipm2[i] = np.concatenate((first_sipm2[i], d[f'a_first_sipm2_thr{i}']))


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

true_t1     = np.array(true_t1)
sipm_t1     = np.array(sipm_t1)
true_t2     = np.array(true_t2)
sipm_t2     = np.array(sipm_t2)
first_sipm1 = np.array(first_sipm1)
first_sipm2 = np.array(first_sipm2)

sns_response1 = np.array(sns_response1)
sns_response2 = np.array(sns_response2)

phot1       = np.array(phot1)
phot2       = np.array(phot2)
phot_like1  = np.array(phot_like1)
phot_like2  = np.array(phot_like2)
true_dep_e1 = np.array(true_dep_e1)
true_dep_e2 = np.array(true_dep_e2)

events = np.array(events)

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

sns_response = np.concatenate((sns_response1, sns_response2))
phot         = np.concatenate((phot1, phot2))
phot_like    = np.concatenate((phot_like1, phot_like2))
true_dep_e   = np.concatenate((true_dep_e1, true_dep_e2))

n_int = len(true_x) # number of interactions

d1 = true_r1 - reco_r1
d2 = true_r2 - reco_r2
diff_r_matrix = np.concatenate((d1, d2))

d1_phi = true_phi1 - reco_phi1
d2_phi = true_phi2 - reco_phi2
diff_phi_matrix = np.concatenate((d1_phi, d2_phi))

d1_z = true_z1 - reco_z1
d2_z = true_z2 - reco_z2
diff_z_matrix = np.concatenate((d1_z, d2_z))


### read sensor positions from database
DataSiPM     = db.DataSiPMsim_only('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

speed_in_vacuum = 0.299792458# * units.mm / units.ps
ave_speed_in_LXe = 0.210 #* units.mm / units.ps


### Positions
pos_1 = np.array([reco_x1, reco_y1, reco_z1]).transpose()
pos_2 = np.array([reco_x2, reco_y2, reco_z2]).transpose()
dist1 = [[] for j in range(4)]
dist2 = [[] for j in range(4)]
for j in range(4):
    sipm_pos_1 = np.array([DataSiPM_idx.loc[first_sipm1[j]].X, DataSiPM_idx.loc[first_sipm1[j]].Y, DataSiPM_idx.loc[first_sipm1[j]].Z]).transpose()
    sipm_pos_2 = np.array([DataSiPM_idx.loc[first_sipm2[j]].X, DataSiPM_idx.loc[first_sipm2[j]].Y, DataSiPM_idx.loc[first_sipm2[j]].Z]).transpose()

    ### Distance of the interaction point from the SiPM seeing the first photon
    dist1[j].append(np.linalg.norm(np.subtract(pos_1, sipm_pos_1), axis=1))
    dist2[j].append(np.linalg.norm(np.subtract(pos_2, sipm_pos_2), axis=1))

diff_t_matrix      = [[] for j in range(4)]
diff_reco_t_matrix = [[] for j in range(4)]
for j in range(4):
    #d1_t = true_t1 - sipm_t1[j]
    #d2_t = true_t2 - sipm_t2[j]
    #diff_t_matrix[j].append((np.concatenate((d1_t, d2_t))).flatten())

    reco_t1   = np.array(sipm_t1[j]) - (np.array(dist1[j])/ave_speed_in_LXe)
    reco_t2   = np.array(sipm_t2[j]) - (np.array(dist2[j])/ave_speed_in_LXe)
    d1_reco_t = true_t1 - reco_t1
    d2_reco_t = true_t2 - reco_t2
    diff_reco_t_matrix[j].append((np.concatenate((d1_reco_t, d2_reco_t))).flatten())


sel_phot_like  = (phot_like == True)
sel_compt_like = (phot_like == False)

print(f'Number of interactions for phot = {len(phot_like[sel_phot_like])}')


print('')
print('**** Phi ****')
print('')

precision = .3 # mm
r_max     = 410

phi_range = (-3.15, 3.15)
phi_width = precision/r_max
phi_bins  = int((phi_range[1] - phi_range[0])/phi_width)

r_range = (380, 410)
r_width = precision
r_bins  = int((r_range[1] - r_range[0])/r_width)

err_range_phot = (-0.01, 0.01)
err_width_phot =  precision/r_max # divide by the maximum radius to have a maximum space of 0.5 mm
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-0.15, 0.15)
err_width_compt =  precision/r_max # divide by the maximum radius to have a maximum space of 0.5 mm
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true phi = {phi_bins}, true r = {r_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')


## photoelectric-like events
h, edges = np.histogramdd((true_phi[sel_phot_like], true_r[sel_phot_like],
                           diff_phi_matrix[sel_phot_like]), bins=(phi_bins, r_bins, err_bins_phot),
                           range=(phi_range, r_range, err_range_phot))

# Save the error matrix for the fast MC.
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
file_name = folder + 'errmat_nest_phi_phot_like.npz'
#print('eff: ', eff, 'xedges: ', xedges, 'yedges: ', yedges, 'zedges: ', zedges, 'xmin: ', xmin, 'ymin: ', ymin, 'zmin: ', zmin, 'dx: ', dx, 'dy: ', dy, 'dz: ', dz)
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Phot-like')
a = np.sum(h, axis=2)
print(a.shape, a.shape[0]*a.shape[1])
print(np.count_nonzero(a))


## compton-like events
h, edges = np.histogramdd((true_phi[sel_compt_like], true_r[sel_compt_like],
                           diff_phi_matrix[sel_compt_like]), bins=(phi_bins, r_bins, err_bins_phot),
                           range=(phi_range, r_range, err_range_compt))


# Save the error matrix for the fast MC.
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
file_name = folder + 'errmat_nest_phi_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Compt-like')
a = np.sum(h, axis=2)
print(a.shape, a.shape[0]*a.shape[1])
print(np.count_nonzero(a))


print('')
print('**** Z ****')
print('')

precision = .3 # mm
r_max = 410

z_range = (-975, 975)
z_width = precision
z_bins  = int((z_range[1] - z_range[0])/z_width)

r_range = (380, 410)
r_width = precision
r_bins  = int((r_range[1] - r_range[0])/r_width)

err_range_phot = (-3, 3)
err_width_phot =  precision
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-100, 100)
err_width_compt =  precision
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true z = {z_bins}, true r = {r_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')

## photoelectric-like events
h, edges = np.histogramdd((true_z[sel_phot_like], true_r[sel_phot_like],
                           diff_z_matrix[sel_phot_like]), bins=(z_bins, r_bins, err_bins_phot),
                           range=(z_range, r_range, err_range_phot))


# Save the error matrix for the fast MC.
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
file_name = folder + 'errmat_nest_z_phot_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Phot-like')
a = np.sum(h, axis=2)
print(a.shape, a.shape[0]*a.shape[1])
print(np.count_nonzero(a))

## compton-like events
h, edges = np.histogramdd((true_z[sel_compt_like], true_r[sel_compt_like],
                           diff_z_matrix[sel_compt_like]), bins=(z_bins, r_bins, err_bins_compt),
                           range=(z_range, r_range, err_range_compt))


# Save the error matrix for the fast MC.
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
file_name = folder + 'errmat_nest_z_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, zmin=zmin, dx=dx, dy=dy, dz=dz)

print('Compt-like')
a = np.sum(h, axis=2)
print(a.shape, a.shape[0]*a.shape[1])
print(np.count_nonzero(a))


print('')
print('**** R ****')
print('')

precision = .3 # mm
r_max = 410

r_range = (380, 410)
r_width = precision
r_bins  = int((r_range[1] - r_range[0])/r_width)

err_range_phot = (-5, 15)
err_width_phot = precision
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-30, 30)
err_width_compt =  precision
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true r = {r_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')

## photoelectric-like events
h, edges = np.histogramdd((true_r       [sel_phot_like],
                           diff_r_matrix[sel_phot_like]), bins=(r_bins, err_bins_phot),
                           range=(r_range, err_range_phot))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_r_phot_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Phot-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))

## compton-like events
h, edges = np.histogramdd((true_r[sel_compt_like],
                           diff_r_matrix[sel_compt_like]), bins=(r_bins, err_bins_compt),
                           range=(r_range, err_range_compt))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_r_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Compt-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))

print('')
print('')
print('**** T ****')

precision = 5 # ps
#r_max    = 410

t_range = (1000, 3500)
t_width = precision
t_bins  = int((t_range[1] - t_range[0])/t_width)


### NO THRESHOLD IN CHARGE:
print('')
print('NO THRESHOLD IN CHARGE:')
err_range_phot = (-400, 100)
err_width_phot = precision
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-500, 300)
err_width_compt =  precision
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true t = {t_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')

## photoelectric-like events
h, edges = np.histogramdd((true_t[sel_phot_like],
                           diff_reco_t_matrix[0][0][sel_phot_like]), bins=(t_bins, err_bins_phot),
                           range=(t_range, err_range_phot))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr0pes_phot_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)


print('Phot-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))

## compton-like events
h, edges = np.histogramdd((true_t[sel_compt_like],
                           diff_reco_t_matrix[0][0][sel_compt_like]), bins=(t_bins, err_bins_compt),
                           range=(t_range, err_range_compt))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr0pes_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Compt-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))



### THR = 0.5 PES IN CHARGE:
print('')
print('THR = 0.5 PES IN CHARGE:')
err_range_phot = (-500, 100)
err_width_phot = precision
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-550, 250)
err_width_compt =  precision
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true t = {t_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')

## photoelectric-like events
h, edges = np.histogramdd((true_t[sel_phot_like],
                           diff_reco_t_matrix[1][0][sel_phot_like]), bins=(t_bins, err_bins_phot),
                          range=(t_range, err_range_phot))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr0.5pes_phot_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)


print('Phot-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))

## compton-like events
h, edges = np.histogramdd((true_t[sel_compt_like],
                           diff_reco_t_matrix[1][0][sel_compt_like]), bins=(t_bins, err_bins_compt),
                          range=(t_range, err_range_compt))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr0.5pes_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Compt-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))



### THR = 1.0 PES IN CHARGE:
print('')
print('THR = 1.0 PES IN CHARGE:')
err_range_phot = (-1400, 100)
err_width_phot = precision
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-1500, 200)
err_width_compt =  precision
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true t = {t_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')

## photoelectric-like events
h, edges = np.histogramdd((true_t[sel_phot_like],
                           diff_reco_t_matrix[2][0][sel_phot_like]), bins=(t_bins, err_bins_phot),
                          range=(t_range, err_range_phot))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr1.0pes_phot_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)


print('Phot-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))

## compton-like events
h, edges = np.histogramdd((true_t[sel_compt_like],
                           diff_reco_t_matrix[2][0][sel_compt_like]), bins=(t_bins, err_bins_compt),
                          range=(t_range, err_range_compt))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr1.0pes_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Compt-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))


### THR = 1.5 PES IN CHARGE:
print('')
print('THR = 1.5 PES IN CHARGE:')
err_range_phot = (-1450, 100)
err_width_phot = precision
err_bins_phot  = int((err_range_phot[1] - err_range_phot[0])/err_width_phot)

err_range_compt = (-1600, 150)
err_width_compt =  precision
err_bins_compt  = int((err_range_compt[1] - err_range_compt[0])/err_width_compt)
print(f'Number bins: true t = {t_bins}, err phot = {err_bins_phot}, err compt = {err_bins_compt}')

## photoelectric-like events
h, edges = np.histogramdd((true_t[sel_phot_like],
                           diff_reco_t_matrix[3][0][sel_phot_like]), bins=(t_bins, err_bins_phot),
                          range=(t_range, err_range_phot))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr1.5pes_phot_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)


print('Phot-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))

## compton-like events
h, edges = np.histogramdd((true_t[sel_compt_like],
                           diff_reco_t_matrix[3][0][sel_compt_like]), bins=(t_bins, err_bins_compt),
                          range=(t_range, err_range_compt))

# Save the error matrix for the fast MC.
eff    = np.array([1])
xedges = edges[0]
yedges = edges[1]
xmin   = xedges[0]; xmin = np.array(xmin)
ymin   = yedges[0]; ymin = np.array(ymin)
dx     = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
dy     = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
file_name = folder + 'errmat_nest_t_thr1.5pes_compt_like.npz'
np.savez(file_name, errmat=h, eff=eff, xmin=xmin, ymin=ymin, dx=dx, dy=dy)

print('Compt-like')
a = np.sum(h, axis=1)
print(a.shape)
print(np.count_nonzero(a))
