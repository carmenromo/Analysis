import os, sys

import pandas as pd
import numpy as np

import antea.database.load_db as db

"""
Input: convoluted signal, format: event_id, sensor_id, charge_data, charge_mc, charge_conv, ToT
Output: two tables, one with filtered events and the other one with sigma_x/y and charge ratio in external corona.
Filters only events with coincidences and max proxy in 4 central SiPMs for Hamamatsu board.
"""

h_centred_sns = [44, 45, 54, 55]
f_centred_sns = [122, 123, 132, 133]

corona   = [11, 12, 13, 14, 15, 16, 17, 18, 21, 31, 41, 51, 61, 71, 81, 82, 83, 84, 85, 86, 87, 88]

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

def sel_max_charge_centre(df, proxy='ToT'):

    h = df[df.sensor_id <= 89]

    if len(h):
        argmax_h = h[proxy].argmax()
        cn_h = h.iloc[argmax_h].sensor_id in h_centred_sns
    else:
        cn_h = False

    return cn_h

def calculate_sigma(df, coord='X', proxy='ToT'):
    sipms_coord = DataSiPM_pb_idx.loc[df.sensor_id][coord].values
    tot     = df[proxy].values

    mean = np.average(sipms_coord, weights=tot)
    var  = np.average((sipms_coord-mean)**2, weights=tot)

    return np.sqrt(var)


def charge_fraction_in_corona(df_h, proxy):
    charge_corona = df_h[df_h.sensor_id.isin(corona)][proxy].sum()
    charge_tot    = df_h[proxy].sum()

    return charge_corona/charge_tot

DataSiPM_pb     = db.DataSiPM('petalo', 0, 'PB')
DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

start = int(sys.argv[1])
numb  = int(sys.argv[2])
proxy = str(sys.argv[3])

folder = '/sim/petbox/mix_HamVUV_FBK_tile5centered_1nsTOF/tot/'
filein = folder + 'petbox_tot_no_fluct.{0}.h5'
out_file  = '/home/rolucar/PetBox/charge_computation/data_charge_computation/petbox_tot_no_fluct_filtered_{0}.{1}_{2}.h5'.format(proxy, start, numb)

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
#    print(sns_response)

    data = sns_response[sns_response.ToT > 0]

    data['tofpet_id'] = data['sensor_id'].apply(tofpetid)
    nsipms = data.groupby(['event_id', 'tofpet_id'])['sensor_id'].nunique()
    nh     = nsipms[:, 0].rename('nsipms0')
    nf     = nsipms[:, 2].rename('nsipms2')
    d1     = data.join(nh, on=['event_id'])
    d2     = d1.join(nf, on=['event_id'])


    coinc_sel  = d2.groupby(['event_id']).apply(select_coincidences)

    data       = d2.set_index('event_id')
    data       = data[coinc_sel]

    charge_sel = data.groupby(['event_id']).apply(sel_max_charge_centre, proxy)
    df_final   = data[charge_sel]
    df_final   = df_final.reset_index()
    dataf.append(df_final)
    print(f'2: {len(df_final)}')

    # resx  = df_final.groupby(['event_id', 'tofpet_id']).apply(calculate_sigma, 'X', proxy)
    # resy  = df_final.groupby(['event_id', 'tofpet_id']).apply(calculate_sigma, 'Y', proxy)
    # ratio = df_final.groupby(['event_id', 'tofpet_id']).apply(charge_fraction_in_corona, proxy)
    #
    # rx = resx.reset_index(name='sigma_x')
    # ry = resy.reset_index(name='sigma_y')
    # dr = ratio.reset_index(name='charge_ratio')
    #
    # rx['sigma_y']      = ry.sigma_y
    # rx['charge_ratio'] = dr.charge_ratio
    # rx.tofpet_id       = rx.tofpet_id.astype('int8')
    #
    # dataa.append(rx)

sel_df  = pd.concat(dataf)
#sel_agg = pd.concat(dataa)

store = pd.HDFStore(out_file, "w", complib=str("zlib"), complevel=4)
store.put('analysis', sel_df, format='table')
#store.put('aggregated', sel_agg, format='table', data_columns=True)
store.close()
