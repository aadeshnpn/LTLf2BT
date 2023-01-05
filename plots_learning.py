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


def get_data(filename='/tmp/learning_30.pickle'):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def plot_data_tracelen_power(success_prob, ax_power, i):
    tracelens = [10, 15, 20, 25, 30, 40, 50]
    # Box plot
    success_prob = [success_prob[data]['success'] for data in success_prob.keys()]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp2 = ax_power.boxplot(
            success_prob, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.3)
    if i ==0:
        ax_power.set_ylabel('Success Probability')
        ax_power.set_xlabel('Propagation Time')
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_xticklabels(tracelens)
    ax_power.set_title(tracelens[i])


def plot_data_uncertainty_discount_power(success_prob, ax_power, i):

    # Box plot
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    discounts = [0.99, 0.95, 0.9, 0.85, 0.8]
    success_prob = [success_prob[data]['success'] for data in success_prob.keys()]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp2 = ax_power.boxplot(
            success_prob, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.3)
    if i ==0:
        ax_power.set_ylabel('Success Probability')
        ax_power.set_xlabel('Discount Factor')
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_xticklabels(discounts)
    ax_power.set_title(uncertainties[i])


def plot_tracelen_vs_static_propstep(
        datas, uncetainty, discount, fname='tracelen_propstep'):
    propsteps = [10, 15, 20, 25, 30, 40, 50]
    tracelens = [10, 15, 20, 25, 30, 40, 50]
    # for data in datas.keys():
    # print(discount, uncetainty, tracelen )
    info = dict()
    for t in tracelens:
        info[t] = dict()
        for p in propsteps:
            info[t][p] = {'trace':[], 'success':[]}
            btstatus = [d[1] for d in datas[discount][uncetainty][t][p]['result'][0]]
            trace = [d[0] for d in datas[discount][uncetainty][t][p]['result'][0]]
            for j in range(0, len(btstatus), 10):
                results = [True if btstatus[j+i] == common.Status.SUCCESS else False for i in range(10)]
                success_prob = sum(results) / 10.0
                # print(j, p, success_prob)
                trace_len = [len(trace[j+i]) for i in range(10)]
                average_len = sum(trace_len) / 10.0
                # print(reward, uc, success_prob, average_len)
                # print(j, p, average_len, success_prob)
                info[t][p]['trace'].append(average_len)
                info[t][p]['success'].append(success_prob)

    fig_power = plt.figure(figsize=(14, 10), dpi=100)
    f = 0
    for t in info.keys():
        ax_power = fig_power.add_subplot(4, 2, f+1)
        plot_data_tracelen_power(info[t], ax_power, f)
        f += 1

    plt.tight_layout(pad=0.5)

    maindir = '/tmp/'
    # fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def plot_discount_uncertainty(
        datas, tracelen=40, propsteps=25, fname='tracelen_discount_uncertain'):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    discounts = [0.99, 0.95, 0.9, 0.85, 0.8]
    t = tracelen
    p = propsteps
    info = dict()
    for u in uncertainties:
        info[u] = dict()
        for d in discounts:
            info[u][d] = {'trace':[], 'success':[]}
            btstatus = [data[1] for data in datas[d][u][t][p]['result'][0]]
            trace = [data[0] for data in datas[d][u][t][p]['result'][0]]
            for j in range(0, len(btstatus), 10):
                results = [True if btstatus[j+i] == common.Status.SUCCESS else False for i in range(10)]
                success_prob = sum(results) / 10.0
                # print(j, p, success_prob)
                trace_len = [len(trace[j+i]) for i in range(10)]
                average_len = sum(trace_len) / 10.0
                # print(reward, uc, success_prob, average_len)
                # print(j, p, average_len, success_prob)
                info[u][d]['trace'].append(average_len)
                info[u][d]['success'].append(success_prob)

    fig_power = plt.figure(figsize=(14, 10), dpi=100)
    f = 0
    for u in info.keys():
        ax_power = fig_power.add_subplot(2, 2, f+1)
        plot_data_uncertainty_discount_power(info[u], ax_power, f)
        f += 1

    plt.tight_layout(pad=0.5)

    maindir = '/tmp/'
    # fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


    # for discount in discounts:
    #     results[discount] = dict()
    #     for uc in uncertainties:
    #         results[discount][uc] = dict()
    #         for tlen in tracelens:
    #             results[discount][uc][tlen] = dict()
    #             for pstep in propsteps:
    #                 results[discount][uc][tlen][pstep] = dict()
    #                 res, policy = run_experiment(
    #                     rewards[j], uc, runs, maxtrace=tlen,
    #                     propsteps=pstep, discount=discount)
    #                 results[discount][uc][tlen][pstep]['result'] = res
    #                 results[discount][uc][tlen][pstep]['policy'] = policy


def main():
    data = get_data()
    print(len(data))
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    discounts = [0.99, 0.95, 0.9, 0.85, 0.8]

    tracelens = [10, 15, 20, 25, 30, 40, 50]
    propsteps = [10, 15, 20, 25, 30, 40, 50]
    runs = 50

    plot_tracelen_vs_static_propstep(data,
        discount=0.9, uncetainty=uncertainties[0])

    plot_discount_uncertainty(data)


if __name__ =='__main__':
    main()