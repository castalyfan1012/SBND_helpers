import sys
import numpy as np
# Set lartpc_mlreco3d path
LARTPC_MLRECO_PATH = "/sdf/home/b/bearc/lartpc_mlreco3d/"
sys.path.append(LARTPC_MLRECO_PATH)

from mlreco.utils.globals import *

def calc_ke(df,ret_key='true_ke'):
    #T = E - m
    engs = df.true_energy_init.values
    pids = df.true_particle_type.values
    Ts = np.full(engs.shape,-1.)
    for i,pid in enumerate(pids):
        if pid == 0: m = 0
        elif pid == 1: m = ELEC_MASS
        elif pid == 2: m = MUON_MASS
        elif pid == 3: m = PION_MASS
        elif pid == 4: m = PROT_MASS
        else: continue
        if engs[i] >= 0: Ts[i] = engs[i] - m
        else: Ts[i] = engs[i] #maintain dummy value
            
    df[ret_key] = Ts
    return df
def get_true_f_vis(df):
    #Return true_f_vis but also handle edge cases
    eng_init = df.true_energy_init.values
    eng_dep  = df.true_particle_sed_depositions_sum.values
    true_f_vis = np.full(eng_dep.shape,-1.)
    for i,dep in enumerate(eng_dep):
        if dep == -1: continue
        if eng_init[i] == -1 or eng_init[i] == 0: continue
        true_f_vis[i] = dep/eng_init[i]
        #if dep/eng_init[i] < 1:
        #    print(i,dep,eng_init[i],dep/eng_init[i])
    df['true_f_vis'] = true_f_vis
    return df
def clean_df(df):
    #There are values that are np.nan, so we will remove these rows
    return df.dropna(subset=['reco_particle_type','true_particle_type'])