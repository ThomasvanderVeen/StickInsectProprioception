import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import matplotlib.pylab as pylab
import matplotlib
import pickle
from tqdm.notebook import tqdm
from tqdm.contrib import itertools
from scipy.ndimage import gaussian_filter1d
from itertools import product
import pandas as pd
import scipy.stats.mstats as zscore
from dtw import dtw
from matplotlib.colors import LogNorm
import import_ipynb
import os


#possible choices for joint_string:

#'nostep_2'
#'nostep_78'
#'step_11'

parameters = {'N_SIMULATIONS': 2,
              'N_HAIRS': 50,
              'joint_string': 'step',
              'pad': 0.3
              }

constants = {'N_ANGLES': 18,
             'N_JOINTS': 3,
             'T_TOTAL': 5,
             'dt': 0.00025,
             }

constants['N_STEPS'] = int(constants['T_TOTAL']/constants['dt'])

custom_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#FFDB58', '#a65628']
custom_markers = ['o', '^', 'v', '*', '+', 'x', 's', 'p']
custom_linestyles = ['-', '--', '-.', ':', '-', '--', ':', '-.']

plot_parameters = {'legend.fontsize': 8,
                   'figure.figsize': (3.75*0.95, 0.95*3.75/1.61803398875),
                   'figure.dpi': 300,
                   'axes.labelsize': 9,
                   'axes.titlesize': 9,
                   'xtick.labelsize': 8,
                   'ytick.labelsize': 8,
                   "font.family": 'Times New Roman'
                   }
matplotlib.pyplot.rcdefaults()

pylab.rcParams.update(plot_parameters)

def safe_divide(numerator, denominator):
    if isinstance(numerator, np.ndarray):
        result = np.zeros_like(numerator, dtype=float)

        # Check for zero values in the denominator
        zero_indices = np.where(denominator == 0)

        # Set the corresponding elements in the result to zero for zero denominators
        result[zero_indices] = 0.0

        # Perform division for non-zero denominators
        non_zero_indices = np.where(denominator != 0)
        result[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]
    else:
        if denominator == 0:
            return 0
        else:
            return numerator/denominator

    return result


def get_firing_rate_ISI(spike_train, dt):
    spike_index = np.where(spike_train == 1)[0]

    if len(spike_index) > 1:
        inter_spike_interval = np.diff(spike_index) * dt
        firing_rate = 1 / inter_spike_interval
        firing_rate = np.append(firing_rate, firing_rate[-1])

        firing_rate = np.concatenate([[0], firing_rate, [0]])
        spike_index = np.concatenate([[0], spike_index, [spike_train.size - 3]])
    else:
        firing_rate = np.array([])

    return firing_rate, spike_index


def get_firing_rate_convolve(spike_train, dt, t=0.5, sigma=3, nan_bool=True):
    n = int(t / dt)

    firing_rate = np.convolve(spike_train.astype(int), np.ones(n), mode='same') / t
    firing_rate = gaussian_filter1d(firing_rate, sigma=sigma)

    if nan_bool:
        firing_rate[firing_rate < 0.000001] = np.nan

    return firing_rate


def get_confusion_matrix(test_array, real_array):
    intersect = test_array + real_array
    difference = test_array - real_array

    true_positive = intersect[intersect > 1.5].size
    false_positive = difference[difference > 0.5].size
    true_negative = intersect[intersect < 0.5].size
    false_negative = difference[difference < -0.5].size

    return true_positive, false_positive, true_negative, false_negative


def matthews_correlation(true_positive, true_negative, false_positive, false_negative):
    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = ((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)) ** 0.5

    MCC = safe_divide(numerator, denominator)
    return MCC


def get_statistics(true_positive, true_negative, false_positive, false_negative):

    P = true_positive + false_negative
    N = false_positive + true_negative
    T = P+N

    MCC = matthews_correlation(true_positive, true_negative, false_positive, false_negative)
    ACC = (true_positive + true_negative) / T
    TPR = true_positive / (true_positive + false_negative)
    TNR = true_negative / (true_negative + false_positive)

    return [true_positive, true_negative, false_positive, false_negative, P, N, T, MCC, ACC, TPR, TNR]

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")