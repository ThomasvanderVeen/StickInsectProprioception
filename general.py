import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import matplotlib.pylab as pylab
import pickle
from tqdm import tqdm
from tqdm.contrib import itertools
from scipy.ndimage import gaussian_filter1d
from itertools import product
import pandas as pd

parameters = {'N_SIMULATIONS': 2,
              'N_HAIRS': 50,
              }

constants = {'N_ANGLES': 18,
             'N_JOINTS': 3,
             'T_TOTAL': 5,
             'dt': 0.001,
             }

constants['N_STEPS'] = int(constants['T_TOTAL']/constants['dt'])

custom_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#FFDB58', '#a65628']
custom_markers = ['o', '^', 'v', '*', '+', 'x', 's', 'p']
custom_linestyles = ['-', '--', '-.', ':', '-', '--', ':', '-.']

plot_parameters = {'legend.fontsize': 10,
                   'figure.figsize': (5.728, 3.54),
                   'figure.dpi': 100,
                   'axes.labelsize': 15,
                   'axes.titlesize': 15,
                   'xtick.labelsize': 10,
                   'ytick.labelsize': 10,
                   }

pylab.rcParams.update(plot_parameters)