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
    rewards = [
        (-0.5, 2, -2), (-0.04, 10, -2),
        (-0.04, 2, -10), (-0.04, 0.1, -2)
    ]
    xrange = range(1, 9)
    # print('plot',success_prob)
    ax_power.scatter(xrange, success_prob, alpha=0.5, s=100, marker="x")
    if i ==0:
        ax_power.set_ylabel('Success Probability', fontsize='x-large')
        ax_power.set_xlabel('Uncertainty', fontsize='x-large' )
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize='large')
    ax_power.set_xticks([2, 4, 6, 8])
    ax_power.set_xticklabels([2, 4, 6, 8], fontsize='large')
    ax_power.set_title(rewards[i], fontsize='x-large')


def plot_success_prob(success_prob, ax_power, i):
    rewards = [
        (-0.5, 2, -2), (-0.04, 10, -2),
        (-0.04, 2, -10), (-0.04, 0.1, -2)
    ]
    xrange = range(1, 9)
    # print('plot',success_prob)
    ax_power.scatter(xrange, success_prob, alpha=0.5, s=100, marker="x")
    if i ==0:
        ax_power.set_ylabel('Success Probability', fontsize='x-large')
        ax_power.set_xlabel('Uncertainty', fontsize='x-large' )
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize='large')
    ax_power.set_xticks([2, 4, 6, 8])
    ax_power.set_xticklabels([2, 4, 6, 8], fontsize='large')
    ax_power.set_title(rewards[i], fontsize='x-large')


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
        ax_eff.set_ylabel('Trace Length', fontsize='x-large')
        ax_eff.set_xlabel('Uncertainty', fontsize='x-large')
    ax_eff.set_yticks([5, 10, 15, 20])
    ax_eff.set_yticklabels([5, 10, 15, 20], fontsize='large')
    ax_eff.set_xticks(range(1,9))
    ax_eff.set_xticklabels(range(1,9), fontsize='large')
    ax_eff.set_title(rewards[i], fontsize='x-large')


def plot_trace_len(trace_len, ax_eff, i):
    rewards = [
        (-0.5, 2, -2), (-0.04, 10, -2),
        (-0.04, 2, -10), (-0.04, 0.1, -2)
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
            trace_len, 0, sym='', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.8)
    if i ==0:
        ax_eff.set_ylabel('Trace Length', fontsize='x-large')
        ax_eff.set_xlabel('Uncertainty', fontsize='x-large')
    ax_eff.set_yticks([10, 20, 30, 40])
    ax_eff.set_yticklabels([10, 20, 30, 40], fontsize='large')
    ax_eff.set_xticks([2,4,6,8])
    ax_eff.set_xticklabels([2,4,6,8], fontsize='large')
    ax_eff.set_title(rewards[i], fontsize='x-large')


def get_data_mdp_reward_uncertainty(filename='/tmp/mdp_30.pickle'):
    with open(filename, 'rb') as file:
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


def plot_mdp_power(info, fname='mpd_power_30'):
    i=0
    fig_power = plt.figure(figsize=(14, 8), dpi=100)
    for reward in info.keys():
        ax_power = fig_power.add_subplot(4, 4, i+1)
        plot_data_reward_power(info[reward]['success'], ax_power, i)
        i += 1

    plt.tight_layout(pad=0.9)

    maindir = '/tmp/'
    # fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def plot_mdp_eff(info, fname='mdp_efficiency_30'):
    i=0
    fig_eff = plt.figure(figsize=(14, 8), dpi=100)
    for reward in info.keys():
        ax_power = fig_eff.add_subplot(4, 4, i+1)
        plot_data_reward_eff(info[reward]['trace'], ax_power, i)
        i += 1
    plt.tight_layout(pad=0.5)
    maindir = '/tmp/'
    # fname = 'mpd_efficiency_30'
    fig_eff.savefig(
        maindir + '/' + fname + '.png')
    plt.close(fig_eff)


def plot_arms_paper(info, fname='cheese_home'):
    fig = plt.figure(figsize=(4, 8), dpi=100)

    rewards = [
        (-0.5, 2, -2), (-0.04, 10, -2),
        (-0.04, 2, -10), (-0.04, 0.1, -2)
    ]
    subplot = [1, 3, 5, 7]
    i=0
    for reward in rewards:
        ax_power = fig.add_subplot(4, 2, subplot[i])
        ax_eff = fig.add_subplot(4, 2, subplot[i]+1)
        plot_success_prob(info[reward]['success'], ax_power, i)
        plot_trace_len(info[reward]['trace'], ax_eff, i)
        i = i+1

    plt.tight_layout(pad=0.5)
    maindir = '/tmp/'
    # fname = 'mpd_efficiency_30'
    fig.savefig(
        maindir + '/' + fname + '.png')
    plt.close(fig)


def main():
    info = get_data_mdp_reward_uncertainty(filename='/tmp/mdp_cheese_home.pickle')
    print(info.keys())
    plot_arms_paper(info, 'cheese_home')
    # plot_mdp_power(info,'cheese_home_power')
    # plot_mdp_eff(info, 'cheese_home_efficiency')

    # info = get_data_mdp_reward_uncertainty(filename='/tmp/mdp_30.pickle')
    # plot_mdp_power(info,'cheese_power')
    # plot_mdp_eff(info, 'cheese_efficiency')


if __name__ =='__main__':
    main()