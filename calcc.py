import sys  # Import the sys module to manipulate Python runtime environment
import numpy as np  # Import the NumPy library for numerical computations

# Set lartpc_mlreco3d path
# LARTPC_MLRECO_PATH = "/sdf/home/b/bearc/lartpc_mlreco3d/"  # Define the path to the lartpc_mlreco3d directory
LARTPC_MLRECO_PATH = "/sdf/home/c/castalyf/spine/"
sys.path.append(LARTPC_MLRECO_PATH)  # Add the path to sys.path, enabling imports from this directory

from spine.utils.globals import *  # Import necessary items from the mlreco.utils.globals module

def calc_efficiency(pred, true, target_label):
    # Calculate efficiency
    # T in R / T
    assert pred.shape == true.shape  # Ensure shapes of pred and true arrays are equal
    pred_mask = pred == target_label  # Create a mask for predicted labels matching the target label
    true_mask = true == target_label  # Create a mask for true labels matching the target label
    if len(true[true_mask]) == 0: return 0  # If no true labels match the target label, return 0
    return len(true[pred_mask & true_mask]) / len(true[true_mask])  # Calculate efficiency

def calc_purity(pred, true, target_label):
    # Calculate purity
    # R in T / R
    assert pred.shape == true.shape  # Ensure shapes of pred and true arrays are equal
    pred_mask = pred == target_label  # Create a mask for predicted labels matching the target label
    true_mask = true == target_label  # Create a mask for true labels matching the target label
    if len(pred[pred_mask]) == 0: return 0  # If no predicted labels match the target label, return 0
    return len(pred[pred_mask & true_mask]) / len(pred[pred_mask])  # Calculate purity

def get_pur_eff_metric(df, key, bins, debug=False):
    # Calculate purity and efficiency metric
    assert key in df.keys(), "key is not in df!"  # Ensure the specified key exists in the DataFrame
    effs = np.zeros((len(PID_LABELS), len(bins) - 1))  # Initialize array for efficiencies
    purs = effs.copy()  # Initialize array for purities
    counts = effs.copy()  # Initialize array for counts

    bincenters = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
    for target in PID_LABELS.keys():  # Iterate over particle labels
        if debug:
            print('*' * 50)
            print(f'Target = {PID_LABELS[target]}')
        for i, c in enumerate(bincenters):  # Iterate over bin centers
            query = f'{key} >= {bins[i]} and {key} < {bins[i + 1]} '  # Define query for DataFrame
            temp_df = df.query(query)  # Filter DataFrame based on query
            if debug:
                print(len(temp_df.query(f'true_particle_type == {target}')), query)

            pred = temp_df.reco_particle_type.values  # Get predicted labels
            true = temp_df.true_particle_type.values  # Get true labels

            effs[target, i] = calc_efficiency(pred, true, target)  # Calculate efficiency
            purs[target, i] = calc_purity(pred, true, target)  # Calculate purity
            counts[target, i] = len(temp_df.query(f'true_particle_type == {target}'))  # Count occurrences

    return effs, purs, bincenters, counts  # Return efficiency, purity, bin centers, and counts

def build_matrix(pred_labels, true_labels, n_classes=5):
    # Build confusion matrix
    assert pred_labels.shape == true_labels.shape  # Ensure shapes of pred_labels and true_labels are equal
    hist = np.zeros((n_classes, n_classes))  # Initialize confusion matrix
    for i, t in enumerate(true_labels):  # Iterate over true labels
        if t == 5: continue  # Skip kaons
        p = pred_labels[i]  # Get predicted label
        if p == -1 or t == -1 or np.isnan(p) or np.isnan(t): continue  # Skip dummy labels
        hist[int(p), int(t)] += 1  # Update confusion matrix
    return hist  # Return confusion matrix

def calc_overall_efficiency(pred_labels, true_labels, n_classes=5):
    # Calculate overall efficiency
    assert pred_labels.shape == true_labels.shape  # Ensure shapes of pred_labels and true_labels are equal
    num = 0  # Initialize numerator for efficiency
    den = 0  # Initialize denominator for efficiency
    for i, t in enumerate(true_labels):  # Iterate over true labels
        p = pred_labels[i]  # Get predicted label
        if p == -1 or t == -1 or np.isnan(p) or np.isnan(t): continue  # Skip dummy labels
        if p == t: num += 1  # Increment numerator if predicted label matches true label
        den += 1  # Increment denominator
    return num / den  # Return overall efficiency

def get_frac_confused(df, key, pid, bins, normalize=True, debug=False):
    # Get fraction of confusion with other particles
    # T classified as reco, same as confusion matrix
    # pid: pid to get fraction confused
    assert key in df.keys(), "key is not in df!"  # Ensure the specified key exists in the DataFrame
    confusion_v = np.zeros((len(PID_LABELS) - 1, len(bins) - 1))  # Initialize array for confusion fractions
    count_v = confusion_v.copy()  # Initialize array for counts

    bincenters = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
    if debug:
        print('*' * 50)
        print(f'Target = {PID_LABELS[pid]}')
    for i, _ in enumerate(bincenters):  # Iterate over bin centers
        query = f'{key} >= {bins[i]} and {key} < {bins[i + 1]} '  # Define query for DataFrame
        temp_df = df.query(query)  # Filter DataFrame based on query
        if debug:
            print(len(temp_df.query(f'true_particle_type == {pid}')), query)

        pred = temp_df.reco_particle_type.values  # Get predicted labels
        true = temp_df.true_particle_type.values  # Get true labels

        hist = build_matrix(pred, true)  # Build confusion matrix
        for j, _ in enumerate(hist):
            if debug: print(f'column {j} : ', hist[:, j])
            count_v[j, i] = np.sum(hist[:, j])  # Update counts
        if debug: print(f'target confusion (pid = {pid}) : ', hist[:, pid])

        confusion = hist[:, pid]
        total = np.sum(confusion)
        if normalize:
            if total > 0: confusion /= total  # Normalize confusion fraction
        confusion_v[:, i] = hist[:, pid]  # Update confusion fractions
    return confusion_v, count_v  # Return confusion fractions and counts

def calc_ke(df, ret_key='true_ke'):
    # Calculate kinetic energy
    # T = E - m
    engs = df.true_energy_init.values  # Get initial energies
    pids = df.true_particle_type.values  # Get particle types
    Ts = np.full(engs.shape, -1.)  # Initialize kinetic energies
    for i, pid in enumerate(pids):  # Iterate over particle types
        if pid == 0: m = 0
        elif pid == 1: m = ELEC_MASS
        elif pid == 2: m = MUON_MASS
        elif pid == 3: m = PION_MASS
        elif pid == 4: m = PROT_MASS
        else: continue
        if engs[i] >= 0: Ts[i] = engs[i] - m  # Calculate kinetic energy
        else: Ts[i] = engs[i]  # Maintain dummy value

    df[ret_key] = Ts  # Add kinetic energy column to DataFrame
    return df

def get_true_f_vis(df):
    # Get true visible energy fraction
    if 'true_ke' not in df.keys():
        df = calc_ke(df)  # Calculate kinetic energy if not already present
    eng_init = df.true_ke.values  # Get initial energies
    eng_dep = df.true_particle_sed_depositions_sum.values  # Get deposited energies
    true_f_vis = np.full(eng_dep.shape, -1.)  # Initialize true visible energy fractions
    for i, dep in enumerate(eng_dep):  # Iterate over deposited energies
        if dep == -1: continue
        if eng_init[i] == -1 or eng_init[i] == 0: continue
        true_f_vis[i] = dep / eng_init[i]  # Calculate true visible energy fraction
    df['true_f_vis'] = true_f_vis  # Add true visible energy fraction column to DataFrame
    return df  # Return modified DataFrame
