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


def plot_data_reward_power(success_prob, ax_power, i):
    rewards = [
        (-0.04, 2, -2), (-0.1, 2, -2),
        (-0.5, 2, -2), (-1, 2, -2),
        (-1.5, 2, -2), (-0.04, 5, -2),
        (-0.04, 10, -2), (-0.04, 1, -2),
        (-0.04, 0.5, -2), (-0.04, 0.1, -2),
        (-0.04, 2, -5), (-0.04, 2, -10),
        (-0.04, 2, -1), (-0.04, 2, -0.5),
        (-0.04, 2, -0.1), (-0.04, 5, -5),
        ]
    xrange = range(1, 9)
    # print('plot',success_prob)
    ax_power.scatter(xrange, success_prob, alpha=0.5, s=5, marker="x")
    if i ==0:
        ax_power.set_ylabel('Success Probability')
        ax_power.set_xlabel('Uncertainty')
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_title(rewards[i])


def plot_data_reward_eff(trace_len, ax_eff, i):
    rewards = [
        (-0.04, 2, -2), (-0.1, 2, -2),
        (-0.5, 2, -2), (-1, 2, -2),
        (-1.5, 2, -2), (-0.04, 5, -2),
        (-0.04, 10, -2), (-0.04, 1, -2),
        (-0.04, 0.5, -2), (-0.04, 0.1, -2),
        (-0.04, 2, -5), (-0.04, 2, -10),
        (-0.04, 2, -1), (-0.04, 2, -0.5),
        (-0.04, 2, -0.1), (-0.04, 5, -5),
        ]
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        (0.7, 0.15, 0.15), (0.6, 0.2, 0.2),
        (0.5, 0.25, 0.25), (0.4, 0.3, 0.3),
        ]

    # Box plot
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp2 = ax_eff.boxplot(
            trace_len, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.8)
    if i ==0:
        ax_eff.set_ylabel('Trace Length')
        ax_eff.set_xlabel('Uncertainty')
    ax_eff.set_title(rewards[i])


def get_data_mdp_reward_uncertainty():
    fig_power = plt.figure(figsize=(14, 8), dpi=100)
    fig_eff = plt.figure(figsize=(10, 6), dpi=100)

    with open('/tmp/mdp_30.pickle', 'rb') as file:
        data = pickle.load(file)
    i = 0
    info = dict()
    for reward in data.keys():
        # print(reward)
        info[reward] = {'trace':[], 'success':[]}
        for uc in data[reward]:
            results = [True if data[reward][uc]['result'][i][1] == common.Status.SUCCESS else False for i in range(512)]
            success_prob = sum(results) / 512.0
            trace_len = [len(data[reward][uc]['result'][i][0]) for i in range(512)]
            average_len = sum(trace_len) / 512.0
            # print(reward, uc, success_prob, average_len)
            info[reward]['trace'].append(trace_len)
            info[reward]['success'].append(success_prob)

        # ax_power = fig_power.add_subplot(4, 4, i+1)
        # ax_eff = fig_eff.add_subplot(4, 4, i+1)
        # plot_data_reward_power(
        #     info[reward]['success'], ax_power, i)

        # plot_data_reward_eff(
        #     info[reward]['trace'], ax_eff, i)
        # i += 1
    return info


def plot_mdp_power(info):
    i=0
    fig_power = plt.figure(figsize=(14, 8), dpi=100)
    for reward in info.keys():
        ax_power = fig_power.add_subplot(4, 4, i+1)
        plot_data_reward_power(info[reward]['success'], ax_power, i)
        i += 1

    plt.tight_layout(pad=0.5)

    maindir = '/tmp/'
    fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def plot_mdp_eff(info):
    i=0
    fig_eff = plt.figure(figsize=(14, 8), dpi=100)
    for reward in info.keys():
        ax_power = fig_eff.add_subplot(4, 4, i+1)
        plot_data_reward_eff(info[reward]['trace'], ax_power, i)
        i += 1
    plt.tight_layout(pad=0.5)
    maindir = '/tmp/'
    fname = 'mpd_efficiency_30'
    fig_eff.savefig(
        maindir + '/' + fname + '.png')
    plt.close(fig_eff)


def main():
    info = get_data_mdp_reward_uncertainty()
    plot_mdp_power(info)
    plot_mdp_eff(info)


if __name__ =='__main__':
    main()