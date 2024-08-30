# Standard libraries
import os
from collections import namedtuple
from itertools import product

# Scientific computing
import numpy as np
import pandas as pd
import scipy.stats.mstats as zscore
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Machine learning and statistics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import tukey_hsd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Miscellaneous
import pickle
from tqdm.notebook import tqdm
from tqdm.contrib import itertools
import import_ipynb

# Parameters that can be modified
parameters = {'N_SIMULATIONS': 78,          # The maximum number of simulations
              'N_HAIRS': 50,                # The number of hairs in a hair row
              'pad': 0.3,                   # Padding of figures (mm)
              'save_data': False,            # Saving or not saving data
              'run_optimization': False      # Running or skipping optimization
              }

# Constants, not subject to modification
constants = {'N_ANGLES': 18,                # Number of angles
             'N_JOINTS': 3,                 # Number of joints
             'T_TOTAL': 5,                  # Total time (nostep)
             'T_TOTAL_STEP': 20,            # Total time (step)
             'dt': 0.00025,                 # Timestep
             'figsize_halfcolumn': (2.6*0.95, 0.95*2.6/1.61803398875),
             }

# Calculate number of timesteps from total time and timestep, "step" is for simulations where the stick insect encounters a stair.
constants['N_STEPS'] = int(constants['T_TOTAL']/constants['dt'])
constants['N_STEPS_STEP'] = int(constants['T_TOTAL_STEP']/constants['dt'])

# Plotting colors, markers and linestyles
custom_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#FFDB58', '#a65628']
custom_markers = ['o', '^', 'v', '*', '+', 'x', 's', 'p']
custom_linestyles = ['-', '--', '-.', ':', '-', '--', ':', '-.']

# Plot parameters
plot_parameters = {'legend.fontsize': 8,
                   'figure.figsize': (3.75*0.98, 0.98*3.75/1.61803398875),
                   'figure.dpi': 300,
                   'axes.labelsize': 9,
                   'axes.titlesize': 9,
                   'xtick.labelsize': 8,
                   'ytick.labelsize': 8,
                   "font.family": 'Arial'
                   }

# Reset and set plot parameters
matplotlib.pyplot.rcdefaults()
pylab.rcParams.update(plot_parameters)

# General functions
def safe_divide(numerator, denominator):
    """
    Safely divides the numerator by the denominator, handling division by zero.

    Parameters:
    - numerator: Can be a scalar or a numpy array.
    - denominator: Can be a scalar or a numpy array.

    Returns:
    - The result of the division. If the denominator is zero, returns 0 in those cases.
    """
    if isinstance(numerator, np.ndarray):
        # Initialize the result array with zeros
        result = np.zeros_like(numerator, dtype=float)

        # Create a boolean mask for non-zero denominators
        non_zero_mask = denominator != 0

        # Perform safe division where the denominator is non-zero
        result[non_zero_mask] = numerator[non_zero_mask] / denominator[non_zero_mask]
    else:
        # Handle the scalar case
        return numerator / denominator if denominator != 0 else 0

    return result


def get_firing_rate_ISI(spike_train, dt):
    """
    Calculate the firing rate from a spike train using inter-spike intervals (ISI).

    Parameters:
    - spike_train: A binary array where 1 indicates a spike and 0 indicates no spike.
    - dt: Time step duration corresponding to the time interval between data points in the spike train.

    Returns:
    - firing_rate: Array of firing rates calculated from the ISIs.
    - spike_index: Indices of the spikes in the spike train.
    """
    # Find indices where spikes occur
    spike_index = np.where(spike_train == 1)[0]

    if len(spike_index) > 1:
        # Calculate inter-spike intervals (ISI), and firing rate = 1/ISI
        inter_spike_interval = np.diff(spike_index) * dt
        firing_rate = 1 / inter_spike_interval

        # Append the last firing rate to maintain array length
        firing_rate = np.append(firing_rate, firing_rate[-1])

        # Add padding to firing_rate and spike_index for alignment
        firing_rate = np.concatenate([[0], firing_rate, [0]])
        spike_index = np.concatenate([[0], spike_index, [spike_train.size - 3]])
    else:
        # Handle cases with 0 or 1 spike
        firing_rate = np.array([])

    return firing_rate, spike_index


def get_firing_rate_convolve(spike_train, dt, t=0.5, sigma=3, nan_bool=True):

    """
    Calculate the firing rate from a spike train by convolving with a moving average window
    and applying a Gaussian filter.

    Parameters:
    - spike_train: A binary array where 1 indicates a spike and 0 indicates no spike.
    - dt: Time step duration corresponding to the time interval between data points in the spike train.
    - t: Time window over which to average the spikes, by default 0.5 seconds.
    - sigma: Standard deviation for Gaussian filter smoothing, by default 3.
    - nan_bool: If True, replace firing rates below a threshold (1e-6) with NaN, by default True.

    Returns:
    - firing_rate: firing rate at each timestep
    """

    # Calculate the number of timesteps in the time window
    n_steps = int(t / dt)

    # Compute the firing rate by convolving the spike train with a window function
    firing_rate = np.convolve(spike_train.astype(int), np.ones(n_steps), mode='same') / t

    # Apply Gaussian filter for smoothing
    firing_rate = gaussian_filter1d(firing_rate, sigma=sigma)

    # Replace small firing rates with NaN if nan_bool is True
    if nan_bool:
        firing_rate[firing_rate < 1e-6] = np.nan

    return firing_rate


def get_confusion_matrix(test_array, real_array):
    """
    Calculate the confusion matrix components: true positives (TP), false positives (FP),
    true negatives (TN), and false negatives (FN) for binary classification.

    Parameters:
    - test_array: A binary array representing the predicted labels (0 or 1).
    - real_array: A binary array representing the actual labels (0 or 1).

    Returns:
    - true_positive: The number of true positives (cases where both test and real are 1).
    - false_positive: The number of false positives (cases where test is 1 and real is 0).
    - true_negative: The number of true negatives (cases where both test and real are 0).
    - false_negative: The number of false negatives (cases where test is 0 and real is 1).
    """

    # Calculate intersections and differences
    intersect = test_array + real_array
    difference = test_array - real_array

    # Confusion matrix components
    true_positive = np.sum(intersect > 1.5)  # both test and real are 1
    false_positive = np.sum(difference > 0.5)  # test is 1, real is 0
    true_negative = np.sum(intersect < 0.5)  # both test and real are 0
    false_negative = np.sum(difference < -0.5)  # test is 0, real is 1

    return true_positive, false_positive, true_negative, false_negative


def matthews_correlation(true_positive, true_negative, false_positive, false_negative):
    """
    Calculate the Matthews Correlation Coefficient (MCC) for binary classification.

    Parameters:
    - true_positive: The number of true positives (TP).
    - true_negative: The number of true negatives (TN).
    - false_positive: The number of false positives (FP).
    - false_negative: The number of false negatives (FN).

    Returns:
    - MCC: The Matthews Correlation Coefficient, a measure of the quality of binary classifications.
           The value ranges from -1 (perfectly wrong) to +1 (perfectly correct), with 0 indicating
           no better than random guessing.
    """

    # Calculate numerator and denominator for MCC
    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = ((true_positive + false_positive) *
                   (true_positive + false_negative) *
                   (true_negative + false_positive) *
                   (true_negative + false_negative)) ** 0.5

    # Safely handle the case where the denominator is zero
    MCC = safe_divide(numerator, denominator)

    return MCC


def get_statistics(true_positive, true_negative, false_positive, false_negative):
    """
    Calculate key classification statistics.

    Parameters:
    - true_positive: The number of true positives (TP).
    - true_negative: The number of true negatives (TN).
    - false_positive: The number of false positives (FP).
    - false_negative: The number of false negatives (FN).

    Returns:
    - statistics: A list containing the following values:
        1. true_positive (TP)
        2. true_negative (TN)
        3. false_positive (FP)
        4. false_negative (FN)
        5. P (Total positives: TP + FN)
        6. N (Total negatives: FP + TN)
        7. T (Total cases: P + N)
        8. MCC (Matthews Correlation Coefficient)
        9. ACC (Accuracy)
       10. TPR (True Positive Rate)
       11. TNR (True Negative Rate)
    """

    # Calculate total positives, negatives, and total cases
    P = true_positive + false_negative
    N = false_positive + true_negative
    T = P + N

    # Calculate key statistics
    MCC = matthews_correlation(true_positive, true_negative, false_positive, false_negative)
    ACC = (true_positive + true_negative) / T
    TPR = true_positive / P if P > 0 else 0  # Avoid division by zero
    TNR = true_negative / N if N > 0 else 0  # Avoid division by zero

    # Return the computed statistics as a list
    return [true_positive, true_negative, false_positive, false_negative, P, N, T, MCC, ACC, TPR, TNR]


def create_folder_if_not_exists(folder_path):
    """
    Create a folder at the specified path if it does not already exist.

    Parameters:
    - folder_path: The path of the folder to create.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def convert_to_bins(old_array, n_bins, minimum=0, sum_bool=False):
    """
    Convert an array into a specified number of bins by reshaping and optionally summing values within each bin.

    Parameters:
    - old_array: The input array to be binned.
    - n_bins: The desired number of bins.
    - minimum: The threshold value used when `sum_bool` is False, by default 0.
    - sum_bool: If True, the values in each bin are summed; if False, values are binarized based on `minimum`.

    Returns:
    - binned_array: The array converted into the specified number of bins.

    Notes:
    - If the array cannot be evenly divided into `n_bins`, the last column is duplicated until it can.
    """
    old_array = np.transpose(old_array)

    while old_array.shape[1] != n_bins:
        try:
            # Attempt to reshape and sum across bins
            old_array = np.sum(old_array.reshape(old_array.shape[0], n_bins, -1), axis=2)
        except ValueError:
            # Duplicate the last column to allow reshaping
            old_array = np.column_stack((old_array, old_array[:, -1]))

    if not sum_bool:
        # Binarize the array based on the minimum threshold
        old_array = (old_array > minimum).astype(int)

    return np.transpose(old_array)


def second_index(arr, extrema):
    """
    Find the index of the second highest or second lowest value in an array.

    Parameters:
    - arr: The input array from which to find the second extrema index.
    - extrema: A string indicating whether to find the 'max' (second highest) or 'min' (second lowest) value.

    Returns:
    - second_extrema_idx: The index of the second highest or second lowest value in the array.
    """
    # Copy the array to avoid modifying the original
    temp_arr = arr.copy()

    if extrema == 'max':
        # Find the index of the maximum value
        highest_idx = np.argmax(arr)
        # Temporarily set the highest value to negative infinity
        temp_arr[highest_idx] = -np.inf
        # Find the index of the second highest value
        second_extrema_idx = np.argmax(temp_arr)

        # Analogous to finding second highest value
    elif extrema == 'min':
        lowest_idx = np.argmin(arr)
        temp_arr[lowest_idx] = np.inf
        second_extrema_idx = np.argmin(temp_arr)

    return second_extrema_idx
