import os, sys

import pandas as pd
import numpy as np

import antea.database.load_db as db
from   antea.utils.map_functions import load_map

from invisible_cities.reco.sensor_functions import charge_fluctuation

"""
Input: convoluted signal, format: event_id, sensor_id, charge_data, charge_mc, ToT
Output: filtered events and reconstructed position.
Filters only events with coincidences and max proxy in 4 central SiPMs for Hamamatsu board.
"""

h_centred_sns = [44, 45, 54, 55]
f_centred_sns = [122, 123, 132, 133]

def tofpetid(sid):
    if sid < 100:
        return 0
    else:
        return 2

def select_coincidences(df, min0=1, min2=1):
    s0 = df[df.sensor_id < 89]
    s2 = df[df.sensor_id > 100]

    ns0 = s0.sensor_id.nunique()
    ns2 = s2.sensor_id.nunique()

    if ns0 >= min0:
        plane1 = True
    else:
        plane1 = False

    if ns2 >= min2:
        plane2 = True
    else:
        plane2 = False

    return plane1 & plane2

def sel_max_charge_centre(df, proxy='charge_conv'):

    h = df[df.sensor_id <= 89]

    if len(h):
        argmax_h = h[proxy].argmax()
        cn_h = h.iloc[argmax_h].sensor_id in h_centred_sns
    else:
        cn_h = False

    return cn_h

peak_pe = 1300
single_pe_sigma = np.sqrt(peak_pe)*0.06

def fluctuate_single_sensor(df, proxy='charge_conv'):
    fluct_tot_charge = charge_fluctuation(df[f'tot_{proxy}'].values[0:1], single_pe_sigma)[0]
    tc               = df[f'tot_{proxy}'].unique()[0]
    charge_ratio     = df[proxy].values/tc
    fluct_charge     = fluct_tot_charge * charge_ratio

    return fluct_charge

def calculate_position(df, thr, datasipm, zmap, proxy='charge_conv'):
    df = df[df[proxy] > thr]
    sipms = datasipm.loc[df.sensor_id]

    pos = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    q   = df[proxy].values
    if q.sum() == 0:
        return [-1000, -1000, -1000]
#    print(df)
    mean_pos = np.average(pos, weights=q, axis=0)
    var_pos  = np.average(np.subtract(pos, mean_pos)**2, weights=q, axis=0)
#   var_x  = np.average((sipms.X.values - mean_pos[0])**2, weights=q)
#   zpos = zmap(var_x).value

    zpos = zmap(var_pos[0]).value
    mean_pos[2] = zpos

    return mean_pos

ztable = '/home/paolafer/analysis/tables/z_var_x_table_pet_box_HamamatsuVUV_det_plane_coinc_plane_cent_no_cut.h5'
Zpos = load_map(ztable,
                group  = "Zpos",
                node   = "f2pes200bins",
                x_name = "Var_x",
                y_name = "Zpos",
                u_name = "ZposUncertainty")

DataSiPM_pb     = db.DataSiPM('petalo', 0, 'PB')
DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

start = int(sys.argv[1])
numb  = int(sys.argv[2])
proxy = str(sys.argv[3])
thr   = float(sys.argv[4])

folder = '/sim/petbox/mix_HamVUV_FBK_tile5centered_1nsTOF/tot/'
filein = folder + 'petbox_tot_no_fluct_2thresholds_65muA.{0}.h5'
out_file  = '/home/paolafer/analysis/petbox/mix_HamVUV_FBK_tile5centered/filtered_and_reco_2thresholds_65muA_mc_{0}_thr{1}.{2}_{3}.h5'.format(proxy, int(thr), start, numb)

dataf, dataa = [], []

for ifile in range(start, start+numb):

    file_name = filein.format(ifile, proxy)
    try:
        sns_response = pd.read_hdf(file_name, 'conv')
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} not found'.format(file_name))
        continue
    except KeyError:
        print('Problem in file {}'.format(file_name))
        continue
    print('Analyzing file {}'.format(file_name))

    # Selects sensors above threshold
    data = sns_response[sns_response.ToT > 0]
    data = sns_response
    # Adds tofpet_id column (useful for groupby)
    data['tofpet_id'] = data['sensor_id'].apply(tofpetid)
    nsipms = data.groupby(['event_id', 'tofpet_id'])['sensor_id'].nunique()
    nh     = nsipms[:, 0].rename('nsipms0')
    nf     = nsipms[:, 2].rename('nsipms2')
    d1     = data.join(nh, on=['event_id'])
    d2     = d1.join(nf, on=['event_id'])

    # Selects events with signal in both planes
    coinc_sel  = d2.groupby(['event_id']).apply(select_coincidences)
    data       = d2.set_index('event_id')
    df_coinc   = data[coinc_sel]

    # Adds a column with the total charge per board per event
    charge    = df_coinc.groupby(['event_id', 'tofpet_id'])[proxy].sum()
    c         = charge.reset_index(name=f'tot_{proxy}')
    df_coinc = df_coinc.merge(c, on=['event_id', 'tofpet_id'])

    # Fluctuates the charge of each sensors according to LXe intrinsic non-linearity
    fc_mc = df_coinc.groupby(['event_id', 'tofpet_id']).apply(fluctuate_single_sensor, proxy)

    fluct_charge = []
    for a in fc_mc.values:
        fluct_charge.extend(a)

    # Adds a column with the fluctuates charge
    df_coinc[f'fluct_{proxy}'] = fluct_charge
    df_coinc                   = df_coinc.set_index('event_id')

    #df_hama = df_coinc[df_coinc.tofpet_id == 0]
    #if len(df_hama) == 0:
    #    continue

    # Selects events with maximum charge in one of the 4 central sensors in Hamamatsu plane
    charge_sel = df_coinc.groupby('event_id').apply(sel_max_charge_centre, proxy)
    df_charge  = df_coinc[charge_sel]
    # Reconstructs the position
    p   = df_charge.groupby(['event_id', 'tofpet_id']).apply(calculate_position, thr, DataSiPM_pb_idx, Zpos, proxy)
    pos = np.asarray(p.values.tolist(), dtype=float)
    pos_df = p.reset_index()

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    evts = pos_df.event_id
    tfpt = pos_df.tofpet_id

    pos_df = pd.DataFrame({'event_id': evts, 'tofpet_id': tfpt,
                           'x': x, 'y': y, 'z': z})

    # Adds the position to the main dataframe
    df_charge = df_charge.merge(pos_df, on=['event_id', 'tofpet_id'])

    # Adds a column with the total fluctuated charge for event, for board
    charge    = df_charge.groupby(['event_id', 'tofpet_id'])[f'fluct_{proxy}'].sum()
    c         = charge.reset_index(name=f'tot_fluct_{proxy}')
    df_charge = df_charge.merge(c, on=['event_id', 'tofpet_id'])
    # Drops the column with the total non-fluctuated charge
    df_charge.drop(columns=f'tot_{proxy}')

    df_tot = df_charge.reset_index()
    df_tot = df_tot.drop(columns="index")
    dataf.append(df_tot)
    print(f'{len(df_tot)}')

sel_df  = pd.concat(dataf)

store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
store.put('analysis', sel_df, format='table')
store.close()


