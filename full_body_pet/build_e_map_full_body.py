import sys
import numpy  as np
import pandas as pd

#from invisible_cities.reco.sensor_functions import charge_fluctuation

import antea.database.load_db       as db
import antea.reco.reco_functions    as rf
import antea.reco.mctrue_functions  as mcf
import antea.mcsim.sensor_functions as snsf

def charge_fluctuation(signal, single_pe_rms):
    """Simulate the fluctuation of the pe before noise addition
    produced by each photoelectron
    """
    if single_pe_rms == 0:
        ## Protection for some versions of numpy etc
        return signal

    ## We need to convert to float to get accuracy here
    sig_fl   = signal.astype(float)
    non_zero = sig_fl > 0
    sigma    = np.sqrt(sig_fl[non_zero]) * single_pe_rms
    sig_fl[non_zero] = np.random.normal(sig_fl[non_zero], sigma)
    ## This fluctuation can't give negative signal
    sig_fl[sig_fl < 0] = 0
    return 

### read sensor positions from database
def apply_charge_fluctuation(sns_df: pd.DataFrame, DataSiPM_idx: pd.DataFrame):
    """
    Apply a fluctuation in the total detected charge, sensor by sensor,
    according to a value read from the database.
    """

    def rand_normal(sig):
        return np.random.normal(0, sig)

    pe_resolution = DataSiPM_idx.Sigma / DataSiPM_idx.adc_to_pes
    pe_resolution = pe_resolution.reset_index().rename(columns={'SensorID': 'sensor_id'})
    sns_df        = sns_df.join(pe_resolution.set_index('sensor_id'), on='sensor_id')
    sns_df.rename(columns={0:'pe_res'}, inplace=True)

    sns_df['charge'] += np.apply_along_axis(rand_normal, 0, sns_df.pe_res)

    columns = ['event_id', 'sensor_id', 'charge']

    return sns_df.loc[sns_df.charge > 0, columns]

def calculate_phi_sigma(pos, q, thr):
    pos_sel, q_sel = rf.sel_coord(pos, q, thr)
    if len(pos_sel) != 0:
        pos_phi = rf.from_cartesian_to_cyl(np.array(pos_sel))[:,1]
        _, var_phi = rf.phi_mean_var(pos_phi, q_sel)
        return np.sqrt(var_phi)
    else:
        return 1.e9


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file' , type = int, help = "first file (inclusive)"     )
    parser.add_argument('n_files'    , type = int, help = "number of files to analize" )
    parser.add_argument('thr_e'      , type = int, help = "threshold in energy"  )
    parser.add_argument('thr_phi'    , type = int, help = "threshold in phi coordinate"  )
    parser.add_argument('events_path',             help = "input files path"           )
    parser.add_argument('file_name'  ,             help = "name of input files"        )
    parser.add_argument('data_path'  ,             help = "output files path"          )
    return parser.parse_args()

arguments  = parse_args(sys.argv)
start      = arguments.first_file
numb       = arguments.n_files
thr_e      = arguments.thr_e
thr_phi    = arguments.thr_phi
eventsPath = arguments.events_path
file_name  = arguments.file_name
data_path  = arguments.data_path

thickness = 3
label = 'P7R410Z1950mm'

print(f'Using database {label}')
DataSiPM     = db.DataSiPMsim_only('petalo', 0, label)
DataSiPM_idx = DataSiPM.set_index('SensorID')

folder = '/home/paolafer/sim/full-body-wires-3cm/'
file_full = folder + 'full_body_wires_3cm.{0}.h5' #{0:03d}
evt_file = f'/home/paolafer/analysis/full-body-wires-3cm/e_map/full_body_wires_3cm_e_map.{start}_{numb}_qthr{thr_e}_phithr{thr_phi}'


phi_sigmas1,    phi_sigmas2    = [], []
touched_sipms1, touched_sipms2 = [], []
charge1,        charge2        = [], []
the_events                     = []

for number in range(start, start+numb):
    #number_str = "{:03d}".format(number)
    #filename  = f"{eventsPath}/{file_name}.{number_str}.pet.h5"
    filename  = f"{eventsPath}/{file_name}.{number}.h5"
    try:
        sns_response = load_mcsns_response(filename)
    except ValueError:
        print(f'File {filename} not found')
        continue
    except OSError:
        print(f'File {filename} not found')
        continue
    except KeyError:
        print(f'No object named MC/waveforms in file {filename}')
        continue
    print(f'Analyzing file {filename}')

    sns_response = apply_charge_fluctuation(sns_response, DataSiPM_idx)
    sns_response = rf.find_SiPMs_over_threshold(sns_response, thr_e)

    events = sns_response.event_id.unique()

    for evt in events:

        evt_sns = sns_response[sns_response.event_id == evt]
        if len(evt_sns) == 0: continue

        max_sns  = evt_sns.loc[evt_sns['charge'].idxmax()].sensor_id
        max_sipm = DataSiPM_idx.loc[max_sns]
        max_pos  = max_sipm[['X', 'Y', 'Z']].values

        sipms = DataSiPM_idx.loc[evt_sns.sensor_id]

        sns_positions = sipms[['X', 'Y', 'Z']].values
        sns_ids       = sipms.index.astype('int64').values
        sns_charges   = evt_sns.charge.values

        sns1, sns2, pos1, pos2, q1, q2 = rf.divide_sipms_in_two_hemispheres(sns_ids, sns_positions, sns_charges, max_pos)

        the_events.append(evt)

        if len(pos1) > 0:
            sigma_phi1 = calculate_phi_sigma(pos1, q1, thr_phi)

            phi_sigmas1   .append(sigma_phi1)
            touched_sipms1.append(len(pos1))
            charge1       .append(sum(q1))

        else:
            phi_sigmas1   .append(1.e9)
            touched_sipms1.append(1.e9)
            charge1       .append(1.e9)

        if len(pos2) > 0:
            sigma_phi2 = calculate_phi_sigma(pos2, q2, thr_phi)

            phi_sigmas2   .append(sigma_phi2)
            touched_sipms2.append(len(pos2))
            charge2       .append(sum(q2))

        else:
            phi_sigmas2   .append(1.e9)
            touched_sipms2.append(1.e9)
            charge2       .append(1.e9)


phi_sigmas1    = np.array(phi_sigmas1)
phi_sigmas2    = np.array(phi_sigmas2)
touched_sipms1 = np.array(touched_sipms1)
touched_sipms2 = np.array(touched_sipms2)
charge1        = np.array(charge1)
charge2        = np.array(charge2)
events         = np.array(the_events)

np.savez(evt_file, phi_sigmas1=phi_sigmas1, phi_sigmas2=phi_sigmas2, touched_sipms1=touched_sipms1, touched_sipms2=touched_sipms2, charge1=charge1, charge2=charge2, events=events)
