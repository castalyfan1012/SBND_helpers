import numpy as np  # Import the NumPy library for numerical computations
import pandas as pd
import matplotlib.pyplot as plt  # Import the matplotlib library for plotting
import yaml, os, sys, re
import calcc  # Import the calcc module for calculations
import plotters  # Import the plotters module for plotting utilities
from constants import *  # Import necessary items from the constants module
import sys  # Import the sys module to manipulate Python runtime environment
from mpl_toolkits.axes_grid1 import make_axes_locatable # Alignment of x-axes.
from scipy.stats import bootstrap # For completeness plot


# Set lartpc_mlreco3d path
# LARTPC_MLRECO_PATH = "/sdf/home/b/bearc/lartpc_mlreco3d/"  # Define the path to the lartpc_mlreco3d directory
LARTPC_MLRECO_PATH = "/sdf/home/c/castalyf/spine/"
sys.path.append(LARTPC_MLRECO_PATH)  # Add the path to sys.path, enabling imports from this directory

from spine.utils.globals import *  # Import necessary items from the mlreco.utils.globals module

def plot_hist_edges(edges, values, errors, label, ax=None, **pltkwargs):
    """
    Make step plot from edges and values
    
    Parameters
    ----------
    edges : array-like
        Edges of histogram bins
    values : array-like
        Values of histogram bins
    errors : array-like or None
        Errors of histogram bins
    label : str
        Label for legend
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, use current axes.
    pltkwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.step()
    
    """
    centers = (edges[1:] + edges[:-1]) / 2  # Calculate bin centers
    if ax is None:
        h = plt.step(edges, list(values) + [values[-1]], where='post', label=label, **pltkwargs)  # Make step plot
        if errors is not None:
            e = plt.errorbar(centers, values, yerr=errors, elinewidth=3, fmt='none', color=h[0].get_color(),
                             alpha=0.6, capsize=7)  # Add error bars
        else:
            e = None
    else:
        h = ax.step(edges, list(values) + [values[-1]], where='post', label=label, **pltkwargs)  # Make step plot
        if errors is not None:
            e = ax.errorbar(centers, values, yerr=errors, elinewidth=3, fmt='none', color=h[0].get_color(),
                             alpha=0.6, capsize=7)  # Add error bars
        else:
            e = None
    return h, e  # Return handles for step plot and error bars

def draw_confusion_matrix_binned(hist, figure_name='', class_names=[], show_counts=True, figsize=(7, 5), norm_ax=0):
    """
    Draw a confusion matrix for binned data
    
    Parameters
    ----------
    hist : array-like
        Confusion matrix data
    figure_name : str, optional
        Name of the figure
    class_names : list, optional
        Names of the classes
    show_counts : bool, optional
        Whether to show counts in cells
    figsize : tuple, optional
        Size of the figure
    norm_ax : int, optional
        Axis along which to normalize
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The created axes
    
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)  # Set transparency of figure

    # Normalize the histogram counts to the total number of entries in each true class bin
    norms = np.sum(hist, axis=norm_ax, keepdims=True)  # Calculate normalization factors
    hist_norm = hist / norms  # Normalize histogram counts

    # Initialize plot, fill
    n_classes = len(hist)  # Get number of classes
    xedges = yedges = -0.5 + np.arange(0, n_classes + 1)  # Define bin edges
    hh = ax.pcolormesh(xedges, yedges, hist_norm, cmap='Blues')  # Create a pseudocolor plot
    for i in range(n_classes):
        for j in range(n_classes):
            label = '{:0.3f}\n({})'.format(hist_norm[i, j], int(hist[i, j])) if show_counts else '{:0.3f}'.format(
                hist_norm[i, j])  # Format label
            ax.text(j, i, label, color="white" if hist_norm[i, j] > 0.5 else "black", ha="center", va="center")  # Add text to cells

    # Set axes style and labels
    ax.set_xlabel('Class label')
    ax.set_ylabel('Class prediction')
    if len(class_names) == n_classes:
        ax.set_xticks(np.arange(n_classes), labels=class_names)  # Set tick labels for x-axis
        ax.set_yticks(np.arange(n_classes), labels=class_names)  # Set tick labels for y-axis
    fig.colorbar(hh)  # Add colorbar

    return fig, ax  # Return the created figure and axes

def draw_frac_confused(df, key, pid, bins, xlabel, pid_inds=[0, 1, 2, 3, 4], add_legend=True, set_style=True,
                        add_label=True, label_locs=['center', 'center'], only_primaries=0, normalize=True, debug=False,
                        sim_label='', **pltkwargs):
    """
    Draw fraction confused plot
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    key : str
        Key for the DataFrame column
    pid : int
        Particle ID
    bins : array-like
        Bin edges
    xlabel : str
        Label for the x-axis
    pid_inds : list, optional
        List of particle IDs
    add_legend : bool, optional
        Whether to add a legend
    set_style : bool, optional
        Whether to set the plot style
    add_label : bool, optional
        Whether to add a label
    label_locs : list, optional
        Locations for adding labels
    only_primaries : int, optional
        Specifies which particles to include (0 for all, 1 for primaries, 2 for secondaries)
    normalize : bool, optional
        Whether to normalize the plot
    debug : bool, optional
        Whether to print debug information
    sim_label : str, optional
        Label for the simulation
    pltkwargs : dict
        Additional keyword arguments to pass to the plotting functions
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes
        The created axes
    
    """
    title = PID_LABELS[pid]  # Get particle label
    if only_primaries == 1:
        df = df[df.true_particle_is_primary]  # Filter DataFrame for primaries
        title += ' (Primaries)'
    elif only_primaries == 2:
        df = df[~df.true_particle_is_primary.values.astype('bool')]  # Filter DataFrame for secondaries
        title += ' (Secondaries)'
    if normalize:
        ylabel = 'Fraction predicted'  # Set y-axis label for normalized plot
    else:
        ylabel = 'Number predicted'  # Set y-axis label for unnormalized plot

    # Calculate confusion fractions
    confusion_v, count_v = calcc.get_frac_confused(df, key, pid, bins, normalize=normalize, debug=debug)

    # Create subplots
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, height_ratios=[2, 1])
    ax2.set_xlabel(xlabel)  # Set x-axis label
    ax.set_ylabel(ylabel)  # Set y-axis label for main plot
    ax2.set_ylabel('True particles')  # Set y-axis label for count plot
    for ind, label in PID_LABELS.items():
        if ind not in pid_inds: continue
        if pid == ind:
            linestyle = '-'  # Use solid line for the target particle
            alpha = 1.  # Set full opacity for the target particle
        else:
            linestyle = '--'  # Use dashed line for other particles
            alpha = 0.3  # Set partial opacity for other particles
        ax2.step(bins[:-1], count_v[ind], where='post', label=label, color=COLORS[ind], linestyle=linestyle,
                 alpha=alpha, **pltkwargs)  # Plot counts
        ax.step(bins[:-1], confusion_v[ind], where='post', label=label, color=COLORS[ind], linestyle=linestyle,
                alpha=alpha, **pltkwargs)  # Plot confusion fractions

    if normalize:
        ax.set_ylim([-0.05, 1.05])  # Set y-axis limits for normalized plot
    ax.set_title(title)  # Set plot title

    if add_legend:
        ax.legend()  # Add legend to main plot
        ax2.legend()  # Add legend to count plot
    if set_style:
        plotters.set_style(ax, legend_size=12, bbox_to_anchor=(1.01, 1))  # Set plot style for main plot
        plotters.set_style(ax2, legend_size=12, bbox_to_anchor=(1.01, 1))  # Set plot style for count plot
    if add_label:
        plotters.add_label(ax, sim_label, where=label_locs[0], alpha=0.6, fontsize=16)  # Add label to main plot
        plotters.add_label(ax2, sim_label, where=label_locs[1], alpha=0.6, fontsize=16)  # Add label to count plot

    return fig, (ax, ax2)  # Return the created figure and axes


# 2025 updates:
PLOT_DIR = f'Plots_mpvmpr/Plots_{plotters.day}'


# def plot_shower_2dhist(csv_file, title, x_label, y_label, column_x, column_y, fontsize=16):
#     from matplotlib.gridspec import GridSpec

#     # Load the CSV data
#     df = pd.read_csv(csv_file)
#     true_ke    = np.array(df[column_x])
#     reco_calo  = np.array(df[column_y]) / 0.761

#     # Fractional error: (reco - true) / true
#     fractional_error = (reco_calo - true_ke) / true_ke

#     # Fixed bins
#     max_limit = 1000
#     num_bins  = 50
#     xbins = np.linspace(0, max_limit, num_bins)
#     ybins = np.linspace(0, max_limit, num_bins)

#     digitized   = np.digitize(true_ke, xbins)
#     bin_centers = 0.5 * (xbins[1:] + xbins[:-1])

#     # ── Layout: two rows share column-0; colorbar lives in column-1 row-0 only ──
#     # This ensures ax1 and ax2 have identical widths and left edges.
#     fig = plt.figure(figsize=(6, 8))
#     gs  = GridSpec(
#         2, 2,
#         height_ratios=[4, 1],
#         width_ratios=[20, 1],
#         hspace=0.05,
#         wspace=0.45,
#     )
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[1, 0])
#     cax = fig.add_subplot(gs[0, 1])

#     # ── Top panel: 2-D histogram ─────────────────────────────────────
#     h = ax1.hist2d(true_ke, reco_calo, bins=[xbins, ybins], cmap='Blues')
#     fig.colorbar(h[3], cax=cax, label='Counts')
#     cax.yaxis.label.set_size(fontsize - 2)
#     cax.tick_params(labelsize=fontsize - 3)

#     ax1.plot([0, max_limit], [0, max_limit], 'r--')
#     ax1.set_xlim([0, max_limit])
#     ax1.set_ylim([0, max_limit])
#     ax1.set_aspect('equal', adjustable='box')

#     # Updated y-axis label (top panel)
#     ax1.set_ylabel(r'$E_{\mathrm{reco}}\ \mathrm{[MeV]}$', fontsize=fontsize)
#     ax1.tick_params(axis='both', labelsize=fontsize - 2)

#     # "SBND Simulation" watermark – bottom-right of 2-D histogram
#     ax1.text(
#         0.97, 0.04, 'SBND Simulation',
#         transform=ax1.transAxes,
#         fontsize=fontsize - 3, color='gray',
#         ha='right', va='bottom',
#     )

#     # ── Bottom panel: fractional-error ratio ─────────────────────────
#     frac_means = [
#         fractional_error[digitized == i].mean()
#         if len(fractional_error[digitized == i]) > 0 else np.nan
#         for i in range(1, len(xbins))
#     ]
#     frac_stds = [
#         fractional_error[digitized == i].std()
#         if len(fractional_error[digitized == i]) > 0 else np.nan
#         for i in range(1, len(xbins))
#     ]

#     ax2.errorbar(bin_centers, frac_means, yerr=frac_stds,
#                  fmt='o', color='red', label='Fractional Error')
#     ax2.axhline(0, color='black', linestyle='--')
#     ax2.set_xlim([0, max_limit])
#     ax2.set_ylim([-1, 1])

#     # Updated axis labels (bottom panel)
#     ax2.set_xlabel(r'$E_{\mathrm{true}}\ \mathrm{[MeV]}$', fontsize=fontsize)
#     ax2.set_ylabel(
#         r'$\dfrac{E_{\mathrm{reco}}-E_{\mathrm{true}}}{E_{\mathrm{true}}}$',
#         fontsize=fontsize,
#     )
#     ax2.tick_params(axis='both', labelsize=fontsize - 2)
#     ax2.grid(True)

#     fname = os.path.splitext(os.path.basename(csv_file))[0]
#     plotters.save_plot(fname=fname, fig=fig, folder_name=PLOT_DIR)
#     plt.show()


# # assuming plotters is your custom saving module
# # import plotters

# def plot_bias_correction(
#     particle_type,
#     x_energy,
#     y_energy,
#     bin_width,
#     csv_file,
#     zoom_low_energy=False,       # default: False → original full range behavior
#     zoom_max_energy=200,         # only used when zoom_low_energy=True
#     zoom_ylim=(0.6, 1.0)         # only used when zoom_low_energy=True
# ):
#     df = pd.read_csv(csv_file)
   
#     # Select energy variables based on particle_type, x_energy, and y_energy
#     if particle_type == 'track':
#         if x_energy == 'reco_calo':
#             base_energy = df['reco_calo_ke']
#             xlabel = 'Reco Calorimetric $E$ [MeV]'
#         elif x_energy == 'true_calo':
#             base_energy = df['true_calo_ke']
#             xlabel = 'True Calorimetric $E$ [MeV]'
#         else: # reco_csda
#             base_energy = df['reco_csda_ke']
#             xlabel = 'Reco CSDA $E$ [MeV]'
       
#         if y_energy == 'reco_calo':
#             calo_energy = df['reco_calo_ke']
#             ylabel = 'Reco Calorimetric $E$'
#         elif y_energy == 'true_calo':
#             calo_energy = df['true_calo_ke']
#             ylabel = 'True Calorimetric $E$'
#         else: # reco_csda
#             calo_energy = df['reco_csda_ke']
#             ylabel = 'Reco CSDA $E$'
#     else: # shower
#         if x_energy == 'reco_calo':
#             base_energy = df['reco_calo_ke']
#             xlabel = 'Reco Calorimetric $E$ [MeV]'
#         elif x_energy == 'true_calo':
#             base_energy = df['true_calo_ke']
#             xlabel = 'True Calorimetric $E$ [MeV]'
#         else: # true_ke
#             base_energy = df['true_ke']
#             xlabel = 'True $E_{init}$ [MeV]'
       
#         if y_energy == 'reco_calo':
#             calo_energy = df['reco_calo_ke']
#             ylabel = 'Reco Calorimetric $E$'
#         elif y_energy == 'true_calo':
#             calo_energy = df['true_calo_ke']
#             ylabel = 'True Calorimetric $E$'
#         else: # true_ke
#             calo_energy = df['true_ke']
#             ylabel = 'True $E_{init}$'
   
#     # Compute ratio
#     ratio = calo_energy / base_energy
   
#     # Remove invalid values
#     valid_data = (base_energy > 0) & (ratio > 0)
#     base_energy, ratio = base_energy[valid_data], ratio[valid_data]
   
#     # ── Apply zoom restriction if requested ──────────────────────────
#     if zoom_low_energy:
#         zoom_mask = (base_energy >= 0) & (base_energy <= zoom_max_energy)
#         base_energy = base_energy[zoom_mask]
#         ratio       = ratio[zoom_mask]
#         xbins = np.arange(0, zoom_max_energy + bin_width, bin_width)
#     else:
#         xbins = np.arange(0, 1000 + bin_width, bin_width)
   
#     bin_centers = 0.5 * (xbins[1:] + xbins[:-1])
   
#     # Bootstrap function for median and confidence intervals
#     def bootstrap_median(data, conf_level=0.68, n_resamples=1000):
#         ci = bootstrap((data,), np.median, confidence_level=conf_level, method='percentile', n_resamples=n_resamples)
#         return ci.confidence_interval.low, ci.confidence_interval.high, np.median(data)
   
#     # Compute medians and errors for each bin
#     medians, lower_errors, upper_errors = [], [], []
#     for i in range(len(xbins) - 1):
#         bin_data = ratio[(base_energy >= xbins[i]) & (base_energy < xbins[i + 1])]
#         if len(bin_data) > 1:
#             lower, upper, median = bootstrap_median(bin_data)
#             medians.append(median)
#             lower_errors.append(median - lower)
#             upper_errors.append(upper - median)
#         else:
#             medians.append(np.nan)
#             lower_errors.append(np.nan)
#             upper_errors.append(np.nan)
   
#     # Filter out NaN values
#     medians, lower_errors, upper_errors = map(np.array, [medians, lower_errors, upper_errors])
#     valid_mask = ~np.isnan(medians)
#     bin_centers_valid = bin_centers[valid_mask]
#     medians_valid = medians[valid_mask]
#     lower_errors_valid = lower_errors[valid_mask]
#     upper_errors_valid = upper_errors[valid_mask]
   
#     print("Medians:", medians_valid.tolist())
   
#     # Extract dataset name for plot title
#     title_name = os.path.splitext(os.path.basename(csv_file))[0].split("_")[-1]
   
#     # Plot ── original layout preserved ───────────────────────────────
#     plt.figure(figsize=(8, 5))
#     plt.hist2d(base_energy, ratio, bins=[xbins, np.linspace(0, 1.5, 50)], cmap='Blues')
#     plt.colorbar(label='Counts')
#     plt.errorbar(bin_centers_valid, medians_valid, yerr=[lower_errors_valid, upper_errors_valid],
#                  fmt='ro', label='Medians (confidence level = 0.68)')
#     plt.axhline(y=np.mean(medians_valid), color='red', linestyle='--',
#                 label=f'Mean = {np.mean(medians_valid):.3f}')
#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel(f'{ylabel} / {xlabel.split(" [")[0]}', fontsize=14)
#     # plt.title(f'Bias correction ({title_name})', fontsize=16)
#     plt.title(f'Bias Correction', fontsize=16)
#     plt.legend(fontsize=14, loc='best', framealpha=0.7)
   
#     # Apply zoom limits only when requested (keeps original auto-range otherwise)
#     if zoom_low_energy:
#         plt.xlim(0, zoom_max_energy)
#         plt.ylim(zoom_ylim)
   
#     plt.tight_layout()
   
#     # Save plot
#     fname = os.path.splitext(os.path.basename(csv_file))[0]
#     zoom_tag = "_zoom" if zoom_low_energy else ""
#     plotters.save_plot(fname=f"completeness_{fname}_bin={bin_width}{zoom_tag}", 
#                        fig=plt.gcf(), folder_name=PLOT_DIR)
   
#     plt.show()

    
# def plot_bias_correction_zoommed(
#     particle_type,
#     x_energy,
#     y_energy,
#     bin_width,
#     csv_file,
#     zoom_low_energy=False,       # default: False → original full range behavior
#     zoom_max_energy=200,         # only used when zoom_low_energy=True
#     zoom_ylim=(0.6, 1.0)         # only used when zoom_low_energy=True
# ):
#     df = pd.read_csv(csv_file)
   
#     # Select energy variables based on particle_type, x_energy, and y_energy
#     if particle_type == 'track':
#         if x_energy == 'reco_calo':
#             base_energy = df['reco_calo_ke']
#             xlabel = 'Reco Calorimetric $E$ [MeV]'
#         elif x_energy == 'true_calo':
#             base_energy = df['true_calo_ke']
#             xlabel = 'True Calorimetric $E$ [MeV]'
#         else: # reco_csda
#             base_energy = df['reco_csda_ke']
#             xlabel = 'Reco CSDA $E$ [MeV]'
       
#         if y_energy == 'reco_calo':
#             calo_energy = df['reco_calo_ke']
#             ylabel = 'Reco Calorimetric $E$'
#         elif y_energy == 'true_calo':
#             calo_energy = df['true_calo_ke']
#             ylabel = 'True Calorimetric $E$'
#         else: # reco_csda
#             calo_energy = df['reco_csda_ke']
#             ylabel = 'Reco CSDA $E$'
#     else: # shower
#         if x_energy == 'reco_calo':
#             base_energy = df['reco_calo_ke']
#             xlabel = 'Reco Calorimetric $E$ [MeV]'
#         elif x_energy == 'true_calo':
#             base_energy = df['true_calo_ke']
#             xlabel = 'True Calorimetric $E$ [MeV]'
#         else: # true_ke
#             base_energy = df['true_ke']
#             xlabel = 'True $E_{init}$ [MeV]'
       
#         if y_energy == 'reco_calo':
#             calo_energy = df['reco_calo_ke']
#             ylabel = 'Reco Calorimetric $E$'
#         elif y_energy == 'true_calo':
#             calo_energy = df['true_calo_ke']
#             ylabel = 'True Calorimetric $E$'
#         else: # true_ke
#             calo_energy = df['true_ke']
#             ylabel = 'True $E_{init}$'
   
#     # Compute ratio
#     ratio = calo_energy / base_energy
   
#     # Remove invalid values
#     valid_data = (base_energy > 0) & (ratio > 0)
#     base_energy, ratio = base_energy[valid_data], ratio[valid_data]
   
#     # ── Always use full range for binning and statistics ─────────────
#     # zoom_low_energy only affects axis limits, NOT the bins or data used,
#     # so the mean line and medians are always computed over the full range.
#     xbins = np.arange(0, 1000 + bin_width, bin_width)
#     bin_centers = 0.5 * (xbins[1:] + xbins[:-1])
   
#     # Bootstrap function for median and confidence intervals
#     def bootstrap_median(data, conf_level=0.68, n_resamples=1000):
#         ci = bootstrap((data,), np.median, confidence_level=conf_level, method='percentile', n_resamples=n_resamples)
#         return ci.confidence_interval.low, ci.confidence_interval.high, np.median(data)
   
#     # Compute medians and errors for each bin
#     medians, lower_errors, upper_errors = [], [], []
#     for i in range(len(xbins) - 1):
#         bin_data = ratio[(base_energy >= xbins[i]) & (base_energy < xbins[i + 1])]
#         if len(bin_data) > 1:
#             lower, upper, median = bootstrap_median(bin_data)
#             medians.append(median)
#             lower_errors.append(median - lower)
#             upper_errors.append(upper - median)
#         else:
#             medians.append(np.nan)
#             lower_errors.append(np.nan)
#             upper_errors.append(np.nan)
   
#     # Filter out NaN values
#     medians, lower_errors, upper_errors = map(np.array, [medians, lower_errors, upper_errors])
#     valid_mask = ~np.isnan(medians)
#     bin_centers_valid = bin_centers[valid_mask]
#     medians_valid = medians[valid_mask]
#     lower_errors_valid = lower_errors[valid_mask]
#     upper_errors_valid = upper_errors[valid_mask]
   
#     print("Medians:", medians_valid.tolist())
   
#     # Extract dataset name for plot title
#     title_name = os.path.splitext(os.path.basename(csv_file))[0].split("_")[-1]
   
#     # Plot ── original layout preserved ───────────────────────────────
#     plt.figure(figsize=(8, 5))
#     plt.hist2d(base_energy, ratio, bins=[xbins, np.linspace(0, 1.5, 50)], cmap='Blues')
#     plt.colorbar(label='Counts')
#     plt.errorbar(bin_centers_valid, medians_valid, yerr=[lower_errors_valid, upper_errors_valid],
#                  fmt='ro', label='Medians (confidence level = 0.68)')
#     plt.axhline(y=np.mean(medians_valid), color='red', linestyle='--',
#                 label=f'Mean = {np.mean(medians_valid):.3f}')
#     plt.xlabel(xlabel, fontsize=14)
#     plt.ylabel(f'{ylabel} / {xlabel.split(" [")[0]}', fontsize=14)
#     # plt.title(f'Bias correction ({title_name})', fontsize=16)
#     plt.title(f'Bias Correction', fontsize=16)
#     plt.legend(fontsize=14, loc='best', framealpha=0.7)
   
#     # ── Zoom: strictly adjust axis limits only, bins/stats unchanged ─
#     if zoom_low_energy:
#         plt.xlim(0, zoom_max_energy)
#         plt.ylim(zoom_ylim)
   
#     plt.tight_layout()
   
#     # Save plot
#     fname = os.path.splitext(os.path.basename(csv_file))[0]
#     zoom_tag = "_zoom" if zoom_low_energy else ""
#     plotters.save_plot(fname=f"completeness_{fname}_bin={bin_width}{zoom_tag}", 
#                        fig=plt.gcf(), folder_name=PLOT_DIR)
   
#     plt.show()


# ========Publication version========


def plot_shower_2dhist(csv_file, title, x_label, y_label, column_x, column_y, fontsize=16):
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MultipleLocator

    df = pd.read_csv(csv_file)
    true_ke   = np.array(df[column_x])
    reco_calo = np.array(df[column_y]) / 0.761

    fractional_error = (reco_calo - true_ke) / true_ke

    max_limit = 200
    num_bins  = 50
    xbins = np.linspace(0, max_limit, num_bins)
    ybins = np.linspace(0, max_limit, num_bins)

    digitized   = np.digitize(true_ke, xbins)
    bin_centers = 0.5 * (xbins[1:] + xbins[:-1])

    fig = plt.figure(figsize=(6, 6))
    
    gs  = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    # gs  = GridSpec(
    #     2, 2,
    #     height_ratios=[4, 1],
    #     width_ratios=[20, 1],
    #     hspace=0.03,   # tight vertical gap between subplots
    #     wspace=0.15,   # tight horizontal gap between hist and colorbar
    # )
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    # cax = fig.add_subplot(gs[0, 1])

    # ── Top: 2-D histogram ───────────────────────────────────────────
    h = ax1.hist2d(true_ke, reco_calo, bins=[xbins, ybins], cmap='Blues')
    fig.colorbar(h[3], cax=cax, label='Counts')
    cax.yaxis.label.set_size(fontsize - 2)
    cax.tick_params(labelsize=fontsize - 3)

    ax1.plot([0, max_limit], [0, max_limit], 'r--')
    ax1.set_xlim([0, max_limit])
    ax1.set_ylim([0, max_limit])
    ax1.set_xlabel(r'$E_{\mathrm{true}}\ \mathrm{[MeV]}$', fontsize=fontsize)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_ylabel(r'$E_{\mathrm{reco}}\ \mathrm{[MeV]}$', fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=fontsize - 2)
    ax1.text(
        0.97, 0.04, 
        r'$\mathbf{SBND}$ Simulation',
        transform=ax1.transAxes,
        fontsize=fontsize - 2, color='black',
        ha='right', va='bottom',
    )

    # # ── Bottom: fractional-error ratio ───────────────────────────────
    # N_MIN = 10          # require at least this many events per bin
    # YLIM  = (-1.0, 1.0)
    
    # frac_means, frac_stds, valid_centers = [], [], []
    # for i in range(1, len(xbins)):
    #     mask = digitized == i
    #     n = mask.sum()
    #     if n < N_MIN:
    #         continue                           # skip sparse / ill-defined bins
    #     fe = fractional_error[mask]
    #     frac_means.append(fe.mean())
    #     frac_stds.append(fe.std())
    #     valid_centers.append(bin_centers[i - 1])
    
    # frac_means   = np.array(frac_means)
    # frac_stds    = np.array(frac_stds)
    # valid_centers = np.array(valid_centers)
    
    # # Clip error bars so they don't extend beyond the drawn ylim
    # yerr_lo = np.minimum(frac_stds, frac_means - YLIM[0])   # don't go below -1
    # yerr_hi = np.minimum(frac_stds, YLIM[1] - frac_means)   # don't go above +1
    # yerr_lo = np.clip(yerr_lo, 0, None)
    # yerr_hi = np.clip(yerr_hi, 0, None)
    
    # ax2.errorbar(valid_centers, frac_means,
    #              yerr=[yerr_lo, yerr_hi],
    #              fmt='o', color='red', label='Fractional Error',
    #              capsize=3)
    # ax2.axhline(0, color='black', linestyle='--')
    # ax2.set_xlim([0, max_limit])
    # ax2.set_ylim(YLIM)
    # ax2.set_xlabel(r'$E_{\mathrm{true}}\ \mathrm{[MeV]}$', fontsize=fontsize)
    # ax2.set_ylabel(
    #     r'$\dfrac{E_{\mathrm{reco}}-E_{\mathrm{true}}}{E_{\mathrm{true}}}$',
    #     fontsize=fontsize,
    # )
    # ax2.tick_params(axis='both', labelsize=fontsize - 2)

    # # Minor ticks at ±0.5 (major ticks are at -1, 0, 1)
    # ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    # ax2.tick_params(axis='y', which='minor', length=4)

    # ax2.grid(True)
    # ax2.grid(True, which='minor', linewidth=0.5)

    plt.tight_layout()

    # ── Snap colorbar height to ax1's *actual* rendered bounds ───────
    # Must call fig.canvas.draw() first so tight_layout has finalised
    # all positions, including the shrinkage from set_aspect('equal').
    fig.canvas.draw()
    ax1_pos = ax1.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, ax1_pos.y0, cax_pos.width, ax1_pos.height])

    fname = os.path.splitext(os.path.basename(csv_file))[0]
    plotters.save_plot(fname=fname, fig=fig, folder_name=PLOT_DIR)
    plt.show()



def _bias_correction_core(
    particle_type, x_energy, y_energy, bin_width, csv_file,
    zoom_low_energy, zoom_max_energy, zoom_ylim,
    full_range_only,          # True → plot_bias_correction_zoommed behaviour
):
    """Return (fig, ax) with all cosmetic changes applied."""
    FONTSIZE = 16

    df = pd.read_csv(csv_file)

    if particle_type == 'track':
        col_map = {
            'reco_calo': ('reco_calo_ke', r'Reco Calorimetric $E$ [MeV]'),
            'true_calo': ('true_calo_ke', r'True Calorimetric $E$ [MeV]'),
        }
        default = ('reco_csda_ke', r'Reco CSDA $E$ [MeV]')
        xcol, _  = col_map.get(x_energy, default)
        ycol, _  = col_map.get(y_energy, default)
    else:  # shower
        x_col_map = {
            'reco_calo': ('reco_calo_ke',),
            'true_calo': ('true_calo_ke',),
        }
        xcol = x_col_map.get(x_energy, ('true_ke',))[0]
        y_col_map = {
            'reco_calo': ('reco_calo_ke',),
            'true_calo': ('true_calo_ke',),
        }
        ycol = y_col_map.get(y_energy, ('true_ke',))[0]

    base_energy  = df[xcol]
    calo_energy  = df[ycol]
    ratio        = calo_energy / base_energy

    valid        = (base_energy > 0) & (ratio > 0)
    base_energy  = base_energy[valid]
    ratio        = ratio[valid]

    if full_range_only:
        xbins = np.arange(0, 1000 + bin_width, bin_width)
    else:
        if zoom_low_energy:
            mask        = (base_energy >= 0) & (base_energy <= zoom_max_energy)
            base_energy = base_energy[mask]
            ratio       = ratio[mask]
            xbins       = np.arange(0, zoom_max_energy + bin_width, bin_width)
        else:
            xbins = np.arange(0, 1000 + bin_width, bin_width)

    bin_centers = 0.5 * (xbins[1:] + xbins[:-1])

    def bootstrap_median(data, conf_level=0.68, n_resamples=1000):
        ci = bootstrap(
            (data,), np.median,
            confidence_level=conf_level,
            method='percentile',
            n_resamples=n_resamples,
        )
        return ci.confidence_interval.low, ci.confidence_interval.high, np.median(data)

    medians, lo_errs, hi_errs = [], [], []
    for i in range(len(xbins) - 1):
        bd = ratio[(base_energy >= xbins[i]) & (base_energy < xbins[i + 1])]
        if len(bd) > 1:
            lo, hi, med = bootstrap_median(bd)
            medians.append(med);  lo_errs.append(med - lo);  hi_errs.append(hi - med)
        else:
            medians.append(np.nan); lo_errs.append(np.nan); hi_errs.append(np.nan)

    medians, lo_errs, hi_errs = map(np.array, [medians, lo_errs, hi_errs])
    ok  = ~np.isnan(medians)
    bcv = bin_centers[ok]
    mv  = medians[ok]
    lv  = lo_errs[ok]
    uv  = hi_errs[ok]

    print("Medians:", mv.tolist())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist2d(base_energy, ratio,
              bins=[xbins, np.linspace(0, 1.5, 50)], cmap='Blues')
    cbar = fig.colorbar(ax.collections[0], ax=ax, label='Counts')
    cbar.ax.tick_params(labelsize=FONTSIZE - 2)
    cbar.set_label('Counts', fontsize=FONTSIZE - 2)

    ax.errorbar(bcv, mv, yerr=[lv, uv],
                fmt='ro', label='Median in True Energy Bin')
    ax.axhline(y=np.mean(mv), color='red', linestyle='--',
               label=f'Overall Mean = {np.mean(mv):.3f}')

    ax.set_xlabel(r'$E_{\mathrm{true}}\ \mathrm{[MeV]}$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$E_{\mathrm{reco}}/E_{\mathrm{true}}$',  fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE - 2)

    ax.legend(fontsize=FONTSIZE - 2, loc='best', framealpha=0.7)

    ax.text(
        0.97, 0.04, 
        r'$\mathbf{SBND}$ Simulation',
        transform=ax.transAxes,
        fontsize=FONTSIZE - 2, color='black',
        ha='right', va='bottom',
    )

    if zoom_low_energy:
        ax.set_xlim(0, zoom_max_energy)
        ax.set_ylim(zoom_ylim)

    plt.tight_layout()
    return fig


def plot_bias_correction(
    particle_type, x_energy, y_energy, bin_width, csv_file,
    zoom_low_energy=False,
    zoom_max_energy=200,
    zoom_ylim=(0.6, 1.0),
):
    fig = _bias_correction_core(
        particle_type, x_energy, y_energy, bin_width, csv_file,
        zoom_low_energy, zoom_max_energy, zoom_ylim,
        full_range_only=False,
    )
    fname    = os.path.splitext(os.path.basename(csv_file))[0]
    zoom_tag = "_zoom" if zoom_low_energy else ""
    plotters.save_plot(
        fname=f"completeness_{fname}_bin={bin_width}{zoom_tag}",
        fig=fig, folder_name=PLOT_DIR,
    )
    plt.show()


def plot_bias_correction_zoommed(
    particle_type, x_energy, y_energy, bin_width, csv_file,
    zoom_low_energy=False,
    zoom_max_energy=200,
    zoom_ylim=(0.6, 1.0),
):
    fig = _bias_correction_core(
        particle_type, x_energy, y_energy, bin_width, csv_file,
        zoom_low_energy, zoom_max_energy, zoom_ylim,
        full_range_only=True,   # bins/stats always over full range
    )
    fname    = os.path.splitext(os.path.basename(csv_file))[0]
    zoom_tag = "_zoom" if zoom_low_energy else ""
    plotters.save_plot(
        fname=f"completeness_{fname}_bin={bin_width}{zoom_tag}",
        fig=fig, folder_name=PLOT_DIR,
    )
    plt.show()
