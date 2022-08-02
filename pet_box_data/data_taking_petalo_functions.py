import argparse
import numpy as np

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file', type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'   , type = int, help = "number of files to analize")
    parser.add_argument('run_no'    , type = int, help = "Run number" )
    parser.add_argument('out_path'  ,             help = "Output path")
    return parser.parse_args()

def parse_args_n_keys(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('first_file', type = int, help = "first file (inclusive)"    )
    parser.add_argument('n_files'   , type = int, help = "number of files to analize")
    parser.add_argument('i_key'     , type = int, help = "first key (inclusive)"     )
    parser.add_argument('n_key'     , type = int, help = "number of keys to analize" )
    parser.add_argument('run_no'    , type = int, help = "Run number" )
    parser.add_argument('out_path'  ,             help = "Output path")
    return parser.parse_args()

def from_ToT_to_pes(x):
    return 9.98597793 * np.exp(x/252.69045094)

def save_df(df0, outfile):
    np.savez(outfile, evt_number=np.array([i[0] for i in df0.index]),        t1=df0.t1,
                      cluster   =np.array([i[1] for i in df0.index]),        t2=df0.t2,
                      ctdaq     =df0.ctdaq,   sensor_id=df0.sensor_id,    efine=df0.efine,   efine_corrected=df0.efine_corrected,
                      ct_data   =df0.ct_data, tofpet_id=df0.tofpet_id,    tfine=df0.tfine,   tfine_corrected=df0.tfine_corrected,
                      tac_id    =df0.tac_id, channel_id=df0.channel_id,  intg_w=df0.intg_w, tcoarse_extended=df0.tcoarse_extended,
                      tcoarse   =df0.tcoarse,   ecoarse=df0.ecoarse, intg_w_ToT=df0.intg_w_ToT,       ToT_pe=df0.ToT_pe)

def save_df_perc(df0, outfile):
    np.savez(outfile, evt_number=np.array([i[0] for i in df0.index]),        t1=df0.t1,
                      cluster   =np.array([i[1] for i in df0.index]),        t2=df0.t2,
                      ctdaq     =df0.ctdaq,   sensor_id=df0.sensor_id,    efine=df0.efine,   efine_corrected=df0.efine_corrected,
                      ct_data   =df0.ct_data, tofpet_id=df0.tofpet_id,    tfine=df0.tfine,   tfine_corrected=df0.tfine_corrected,
                      tac_id    =df0.tac_id, channel_id=df0.channel_id,  intg_w=df0.intg_w, tcoarse_extended=df0.tcoarse_extended,
                      tcoarse   =df0.tcoarse,   ecoarse=df0.ecoarse, intg_w_ToT=df0.intg_w_ToT,       ToT_pe=df0.ToT_pe,
                      perc_cor  =df0.perc_cor)
