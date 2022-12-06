import pickle
from py_trees import common, blackboard

import os
import numpy as np
# import scipy.stats as stats
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
import glob
import pathlib


def plot_data_reward_uc(reward, uc, success_prob, trace_len, ax_power, ax_eff, i):
    # print(reward, uc, success_prob)
    success_prob[0], success_prob[1] = success_prob[1], success_prob[0]
    trace_len[0], trace_len[1] = trace_len[1], trace_len[0]
    xrange = range(1, 9)
    # print('plot',success_prob)
    ax_power.scatter(xrange, success_prob, alpha=0.5, s=5, marker="x")
    if i ==1:
        ax_power.set_ylabel('Success Probability')
        ax_power.set_xlabel('Uncertainty')
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # Box plot
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp2 = ax_eff.boxplot(
            trace_len, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.8)


def plot_mdp_reward_uncertainty():
    fig_power = plt.figure(figsize=(14, 8), dpi=100)
    fig_eff = plt.figure(figsize=(10, 6), dpi=100)

    with open('/tmp/mdp.pickle', 'rb') as file:
        data = pickle.load(file)
    i = 0
    info = dict()
    for reward in data.keys():
        trace_info = []
        success_info = []
        # info[reward] = dict()
        info[reward] = {'trace':[], 'success':[]}
        for uc in data[reward]:
            results = [True if data[reward][uc]['result'][i][1] == common.Status.SUCCESS else False for i in range(512)]
            success_prob = sum(results) / 512.0
            trace_len = [len(data[reward][uc]['result'][i][0]) for i in range(512)]
            average_len = sum(trace_len) / 512.0
            print(reward, uc, success_prob, average_len)
            info[reward]['trace'].append(trace_len)
            info[reward]['success'].append(success_prob)
        i += 1
        if i >16:
            break
        ax_power = fig_power.add_subplot(4, 4, i)
        ax_eff = fig_eff.add_subplot(4, 4, i)
        plot_data_reward_uc(
            reward, uc, info[reward]['success'], info[reward]['trace'], ax_power, ax_eff, i)

    plt.tight_layout()
    maindir = '/tmp/'
    fname = 'mpd_power_'

    fig_power.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101
    plt.close(fig_power)

    fname = 'mpd_efficiency_'
    fig_eff.savefig(
        maindir + '/' + fname + '.png')
    plt.close(fig_eff)


def main():
    plot_mdp_reward_uncertainty()


if __name__ =='__main__':
    main()