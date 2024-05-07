import numpy as np
import pandas as pd
import arviz
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
matplotlib.rcParams.update({'font.size': 16, 'text.usetex': True, 'font.family': 'Times New Roman'})

purple = '#4635CE'
orange = 'orange'

kwargs = dict(bins=40, smooth=1, label_kwargs=dict(fontsize=16),
              title_kwargs=dict(fontsize=16), color=purple,
              quantiles=None, levels=(0.68, 0.95), truth_color='k',
              plot_density=False, plot_datapoints=False, fill_contours=False,
              max_n_ticks=3, hist_kwargs={'density': True, 'color':purple})

labels_dict = {
    'M_c': r'$\mathcal{M}_c \ [M_\odot]$',
    'eta': r'$\eta$',
    's1_z': r'$\chi_{\rm 1z}$',
    's2_z': r'$\chi_{\rm 2z}$',
    'd_L': r'$d_{\rm L} \ [{\rm Mpc}]$',
    'ra': r'$\alpha \ [{\rm rad}]$',
    'dec': r'$\delta \ [{\rm rad}]$',
    'iota': r'$\iota \ [{\rm rad}]$',
    'psi': r'$\psi \ [{\rm rad}]$',
    'phase_c': r'$\phi_{\rm c} \ [{\rm rad}]$',
    't_c': r'$t_{\rm c} \ [{\rm s}]$'
    }

def medErrorText(x):
    med = np.median(x)
    low, high = arviz.hdi(x, 0.95)
    lowError = med - low
    highError = high - med

    med = str(round(med, 2))
    lowError = str(round(lowError, 2))
    highError = str(round(highError, 2))

    med_split = med.split('.')
    lowError_split = lowError.split('.')
    highError_split = highError.split('.')

    if len(med_split[1]) == 1:
        med += '0'

    if len(lowError_split[1]) == 1:
        lowError += '0'

    if len(highError_split[1]) == 1:
        highError += '0'

    text = '$' + med + '^{+' + highError + '}' + '_{-' + lowError + '}$'
    return text

def corner_plot(sample_dict, true_params, output_path):
    plotting_samples = []
    labels = []
    truths = []
    for key in sample_dict:
        plotting_samples.append(sample_dict[key])
        labels.append(labels_dict[key])
        truths.append(true_params[key])
    plotting_samples = np.array(plotting_samples).T

    pltidx = np.random.randint(10000)
    plt.figure(pltidx)
    corner.corner(plotting_samples, labels=labels, truths=truths, **kwargs)
    plt.savefig(f'{output_path}/corner_plot.pdf', bbox_inches='tight')
