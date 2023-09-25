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


def get_data(filename='/tmp/learning_30_all.pickle'):
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
    # discounts = [0.99, 0.95, 0.9, 0.85, 0.8]
    discounts = [0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
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
            for r in range(len(datas[discount][uncetainty][t][p]['result'])):
                btstatus = [d[1] for d in datas[discount][uncetainty][t][p]['result'][r]]
                trace = [d[0] for d in datas[discount][uncetainty][t][p]['result'][r]]
                # print(t, p, len(trace), btstatus)
                results = [True if status == common.Status.SUCCESS else False for status in btstatus]
                success_prob = round(sum(results) / p, 2)
                trace_len = [len(t) for t in trace]
                average_len = round(sum(trace_len) / p, 2)
                # print(t, p, success_prob, average_len)
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
        maindir + '/' + fname + '_' + str(discount)+'_.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def plot_discount_uncertainty(
        datas, tracelen=15, propsteps=25,
        fname='tracelen_discount_uncertain',retval=False):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    # discounts = [
    #     0.99, 0.95, 0.9, 0.85, 0.8, 0.7,
    #     0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    discounts = [0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
    t = tracelen
    p = propsteps
    info = dict()
    for u in uncertainties:
        info[u] = dict()
        for d in discounts:
            info[u][d] = {'trace':[], 'success':[]}
            for r in range(len(datas[d][u][t][p]['result'])):
                btstatus = [d[1] for d in datas[d][u][t][p]['result'][r]]
                trace = [d[0] for d in datas[d][u][t][p]['result'][r]]
                # print(t, p, len(trace), btstatus)
                results = [True if status == common.Status.SUCCESS else False for status in btstatus]
                success_prob = round(sum(results) / p, 2)
                trace_len = [len(t) for t in trace]
                average_len = round(sum(trace_len) / p, 2)
                # print(t, p, success_prob, average_len)
                # print(j, p, average_len, success_prob)
                info[u][d]['trace'].append(average_len)
                info[u][d]['success'].append(success_prob)

            # btstatus = [data[1] for data in datas[d][u][t][p]['result'][0]]
            # trace = [data[0] for data in datas[d][u][t][p]['result'][0]]
            # for j in range(0, len(btstatus), 10):
            #     results = [True if btstatus[j+i] == common.Status.SUCCESS else False for i in range(10)]
            #     success_prob = sum(results) / 10.0
            #     # print(j, p, success_prob)
            #     trace_len = [len(trace[j+i]) for i in range(10)]
            #     average_len = sum(trace_len) / 10.0
            #     # print(reward, uc, success_prob, average_len)
            #     # print(j, p, average_len, success_prob)
            #     info[u][d]['trace'].append(average_len)
            #     info[u][d]['success'].append(success_prob)

    if retval:
        return info
    else:
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
            maindir + '/' + fname + '_'+ str(t)+ '_'+ str(p)+ '.png')
        # pylint: disable = E1101
        plt.close(fig_power)


def plot_power(
        datas, tracelen=15, propsteps=25, discount=0.7,
        fname='resilence_power', retval=False):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    t = tracelen
    p = propsteps
    d = discount
    info = dict()
    for u in uncertainties:
        info[u] = {'trace':[], 'success':[]}
        for r in range(len(datas[d][u][t][p]['result'])):
            btstatus = [d[1] for d in datas[d][u][t][p]['result'][r]]
            trace = [d[0] for d in datas[d][u][t][p]['result'][r]]
            # print(t, p, len(trace), btstatus)
            results = [True if status == common.Status.SUCCESS else False for status in btstatus]
            success_prob = round(sum(results) / p, 2)
            trace_len = [len(t) for t in trace]
            average_len = round(sum(trace_len) / p, 2)
            # print(t, p, success_prob, average_len)
            # print(j, p, average_len, success_prob)
            info[u]['trace'].append(average_len)
            info[u]['success'].append(success_prob)

    if retval:
        return info
    else:
        fig_power = plt.figure(figsize=(8, 6), dpi=100)
        ax_power = fig_power.add_subplot(1, 1, 1)

        success_probs = [info[data]['success'] for data in info.keys()]
        medianprops = dict(linewidth=2.5, color='firebrick')
        meanprops = dict(linewidth=2.5, color='#ff7f0e')
        bp2 = ax_power.boxplot(
                success_probs, 0, 'gD', showmeans=True, meanline=True,
                patch_artist=True, medianprops=medianprops,
                meanprops=meanprops, widths=0.3)
        ax_power.set_ylabel('Success Probability', fontsize='large' )
        ax_power.set_xlabel('Uncertainty', fontsize='large')

        ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_power.set_xticklabels(uncertainties)
        ax_power.set_title('Resiliency Power')


        plt.tight_layout(pad=0.5)

        maindir = '/tmp/'
        # fname = 'mpd_power_30'

        fig_power.savefig(
            maindir + '/' + fname + '_'+ str(t)+ '_'+ str(p)+ '.png')
        # pylint: disable = E1101
        plt.close(fig_power)


def plot_efficiency(
        datas, tracelen=15, propsteps=25, discount=0.7,
        fname='resilence_efficiency', retval=False):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    t = tracelen
    p = propsteps
    d = discount
    info = dict()
    for u in uncertainties:
        info[u] = {'trace':[], 'success':[]}
        for r in range(len(datas[d][u][t][p]['result'])):
            btstatus = [d[1] for d in datas[d][u][t][p]['result'][r]]
            trace = [d[0] for d in datas[d][u][t][p]['result'][r]]
            # print(t, p, len(trace), btstatus)
            results = [True if status == common.Status.SUCCESS else False for status in btstatus]
            success_prob = round(sum(results) / p, 2)
            trace_len = [len(t) for t in trace]
            average_len = round(sum(trace_len) / p, 2)
            # print(t, p, success_prob, average_len)
            # print(j, p, average_len, success_prob)
            info[u]['trace'].append(average_len)
            info[u]['success'].append(success_prob)

    if retval:
        return info
    else:
        fig_power = plt.figure(figsize=(8, 6), dpi=100)
        ax_power = fig_power.add_subplot(1, 1, 1)

        trace_lens = [info[data]['trace'] for data in info.keys()]
        medianprops = dict(linewidth=2.5, color='firebrick')
        meanprops = dict(linewidth=2.5, color='#ff7f0e')
        bp2 = ax_power.boxplot(
                trace_lens, 0, 'gD', showmeans=True, meanline=True,
                patch_artist=True, medianprops=medianprops,
                meanprops=meanprops, widths=0.3)
        ax_power.set_ylabel('Trace Length', fontsize='large' )
        ax_power.set_xlabel('Uncertainty', fontsize='large')

        ax_power.set_yticks([10,20,30,40,50])
        ax_power.set_xticklabels(uncertainties)
        ax_power.set_title('Resiliency Efficiency')

        plt.tight_layout(pad=0.5)

        maindir = '/tmp/'
        # fname = 'mpd_power_30'

        fig_power.savefig(
            maindir + '/' + fname + '_'+ str(t)+ '_'+ str(p)+ '.png')
        # pylint: disable = E1101
        plt.close(fig_power)


def plot_power_policy_random_startloc(
        datas, tracelen=15, propsteps=25, discount=0.7,
        fname='mission_power_randomloc'):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    t = tracelen
    p = propsteps
    d = discount
    info = dict()
    for u in uncertainties:
        info[u] = {'trace':[], 'success':[]}
        for j in range(len(datas[u])):
            results = [d[j][0] for d in datas[u]]
            # print(j, [d[j][0] for d in datas[u]])
            # btstatus = [d[1] for d in datas[u][0]]
            trace = [d[j][1] for d in datas[u]]
            success_prob = round(sum(results) / len(results), 2)
            trace_len = [len(t) for t in trace]
            average_len = round(sum(trace_len) / len(results), 2)
            # print(j, u, success_prob, average_len)
            info[u]['trace'].append(average_len)
            info[u]['success'].append(success_prob)

    fig_power = plt.figure(figsize=(8, 6), dpi=100)
    ax_power = fig_power.add_subplot(1, 1, 1)

    success_probs = [info[data]['success'] for data in info.keys()]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp2 = ax_power.boxplot(
            success_probs, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.3)
    ax_power.set_ylabel('Success Probability', fontsize='large' )
    ax_power.set_xlabel('Uncertainty', fontsize='large')

    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_xticklabels(uncertainties)
    ax_power.set_title('Resiliency Power (Random Start Loc)')


    plt.tight_layout(pad=0.5)

    maindir = '/tmp/'
    # fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '_'+ str(t)+ '_'+ str(p)+ '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def plot_efficiency_random_start_loc(
        datas, tracelen=15, propsteps=25, discount=0.7,
        fname='mission_eff_random_loc'):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    t = tracelen
    p = propsteps
    d = discount
    info = dict()
    for u in uncertainties:
        info[u] = {'trace':[], 'success':[]}
        for j in range(len(datas[u])):
            results = [d[j][0] for d in datas[u]]
            # print(j, [d[j][0] for d in datas[u]])
            # btstatus = [d[1] for d in datas[u][0]]
            trace = [d[j][1] for d in datas[u]]
            success_prob = round(sum(results) / len(results), 2)
            trace_len = [len(t) for t in trace]
            average_len = round(sum(trace_len) / len(results), 2)
            # print(j, u, success_prob, average_len)
            info[u]['trace'].append(average_len)
            info[u]['success'].append(success_prob)

    fig_power = plt.figure(figsize=(8, 6), dpi=100)
    ax_power = fig_power.add_subplot(1, 1, 1)

    trace_lens = [info[data]['trace'] for data in info.keys()]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp2 = ax_power.boxplot(
            trace_lens, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.3)
    ax_power.set_ylabel('Trace Length', fontsize='large' )
    ax_power.set_xlabel('Uncertainty', fontsize='large')

    ax_power.set_yticks([3, 6, 9, 12, 15])
    ax_power.set_xticklabels(uncertainties)
    ax_power.set_title('Resiliency Efficiency (Random Start Loc)')

    plt.tight_layout(pad=0.5)

    maindir = '/tmp/'
    # fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '_'+ str(t)+ '_'+ str(p)+ '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def plot_power_cobined(
        datas, datasr, tracelen=15, propsteps=25, discount=0.7,
        fname='resilence_power_combined',ax=None):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    t = tracelen
    p = propsteps
    d = discount
    # For Random Starting Loc
    infor = dict()
    for u in uncertainties:
        infor[u] = {'trace':[], 'success':[]}
        for j in range(len(datasr[u])):
            results = [d[j][0] for d in datasr[u]]
            trace = [d[j][1] for d in datasr[u]]
            success_prob = round(sum(results) / len(results), 2)
            trace_len = [len(t) for t in trace]
            average_len = round(sum(trace_len) / len(results), 2)
            infor[u]['trace'].append(average_len)
            infor[u]['success'].append(success_prob)

    # For Learning
    info = plot_power(
        datas, tracelen=tracelen, propsteps=propsteps, discount=discount,
        fname='resilence_power', retval=True)

    colordict = {
        0: 'gold',
        1: 'olivedrab',
        2: 'orchid',
        3: 'peru',
        4: 'linen',
        5: 'indianred',
        6: 'tomato'}

    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11]
        ]
    ax_power = ax

    success_probs_rand = [infor[data]['success'] for data in infor.keys()]
    success_probs = [info[data]['success'] for data in info.keys()]
    print(len(success_probs_rand), len(success_probs))
    datas = [
        [success_probs[0], success_probs_rand[0]],
        [success_probs[1], success_probs_rand[1]],
        [success_probs[2], success_probs_rand[2]],
        [success_probs[3], success_probs_rand[3]]
    ]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    for j in range(len(positions)):
        bp2 = ax_power.boxplot(
                datas[j], 0, '', showmeans=True, meanline=True,
                patch_artist=True, medianprops=medianprops,
                meanprops=meanprops, widths=0.8, positions=positions[j])
        for patch, color in zip(bp2['boxes'], colordict.values()):
            patch.set_facecolor(color)

    ax_power.legend(
        zip(bp2['boxes']), ['Learning', 'Inference'], fontsize="large", loc="lower left")
    ax_power.set_ylabel('Success Probability', fontsize='x-large' )
    # ax_power.set_xlabel('Uncertainty', fontsize='large')
    ax_power.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_power.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize='large')
    ax_power.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax_power.set_xticklabels([0.95, 0.9, 0.8, 0.7], fontsize='large')
    # ax_power.set_title('Resiliency Power')

    # plt.tight_layout(pad=0.5)

    # maindir = '/tmp/'
    # # fname = 'mpd_power_30'

    # fig_power.savefig(
    #     maindir + '/' + fname + '_'+ str(t)+ '_'+ str(p)+ '.png')
    # # pylint: disable = E1101
    # plt.close(fig_power)


def plot_efficiency_combined(
        datas, datasr, tracelen=15, propsteps=25, discount=0.7,
        fname='resilence_efficiency_combined',ax=None):
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    t = tracelen
    p = propsteps
    d = discount
    infor = dict()
    for u in uncertainties:
        infor[u] = {'trace':[], 'success':[]}
        for j in range(len(datasr[u])):
            results = [d[j][0] for d in datasr[u]]
            # print(j, [d[j][0] for d in datas[u]])
            # btstatus = [d[1] for d in datas[u][0]]
            trace = [d[j][1] for d in datasr[u]]
            success_prob = round(sum(results) / len(results), 2)
            trace_len = [len(t) for t in trace]
            average_len = round(sum(trace_len) / len(results), 2)
            # print(j, u, success_prob, average_len)
            infor[u]['trace'].append(average_len)
            infor[u]['success'].append(success_prob)

    info = plot_efficiency(
        datas, tracelen=tracelen, propsteps=propsteps, discount=discount,
        fname='resilence_efficiency', retval=True)
    colordict = {
        0: 'gold',
        1: 'olivedrab',
        2: 'orchid',
        3: 'peru',
        4: 'linen',
        5: 'indianred',
        6: 'tomato'}

    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11]
        ]

    # fig_power = plt.figure(figsize=(8, 6), dpi=100)
    # ax_power = fig_power.add_subplot(1, 1, 1)
    ax_power = ax

    trace_lens = [info[data]['trace'] for data in info.keys()]
    trace_lens_rand = [infor[data]['trace'] for data in infor.keys()]

    datas = [
        [trace_lens[0], trace_lens_rand[0]],
        [trace_lens[1], trace_lens_rand[1]],
        [trace_lens[2], trace_lens_rand[2]],
        [trace_lens[3], trace_lens_rand[3]]
    ]

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    for j in range(len(positions)):
        bp2 = ax_power.boxplot(
                datas[j], 0, '', showmeans=True, meanline=True,
                patch_artist=True, medianprops=medianprops,
                meanprops=meanprops, widths=0.8, positions=positions[j])
        for patch, color in zip(bp2['boxes'], colordict.values()):
            patch.set_facecolor(color)

    # ax_power.legend(
    #     zip(bp2['boxes']), ['Learning Efficiency', 'Inference Efficiency'],
    #     fontsize="large", loc="upper left", title='Efficiency Metric')
    ax_power.set_ylabel('Trace Length', fontsize='x-large' )
    # ax_power.set_xlabel('Uncertainty', fontsize='large')

    ax_power.set_yticks([10,20,30,40,50])
    ax_power.set_yticklabels([10,20,30,40,50], fontsize='large')
    ax_power.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax_power.set_xticklabels([0.95, 0.9, 0.8, 0.7], fontsize='large')
    # ax_power.set_title('Resiliency Efficiency')

    # plt.tight_layout(pad=0.5)

    # maindir = '/tmp/'
    # # fname = 'mpd_power_30'

    # fig_power.savefig(
    #     maindir + '/' + fname + '.png')
    # # pylint: disable = E1101
    # plt.close(fig_power)


def combine_plots(data, data_rand, fname='learning_cheese_home'):
    fig_power = plt.figure(figsize=(6, 4), dpi=100)
    ax_power = fig_power.add_subplot(1, 2, 1)
    ax_eff = fig_power.add_subplot(1, 2, 2)
    ax_power.sharex =True
    # ax_power.sharey =True
    ax_eff.sharex = True
    plot_power_cobined(
        data, data_rand, tracelen=50, propsteps=200, discount=0.9, ax=ax_power)
    plot_efficiency_combined(
        data, data_rand, tracelen=50, propsteps=200, discount=0.9, ax=ax_eff)

    plt.xlabel(
            'Intended Action Probablity',
            fontsize='x-large', loc="right")

    plt.tight_layout(pad=0.5)
    maindir = '/tmp/'
    # fname = 'mpd_power_30'

    fig_power.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101
    plt.close(fig_power)


def main():
    # data = get_data()
    # print(len(data))
    uncertainties = [
        (0.95, 0.025, 0.025), (0.9, 0.05, 0.05),
        (0.85, 0.075, 0.075), (0.8, 0.1, 0.1),
        ]
    # discounts = [
    #     0.99, 0.95, 0.9, 0.85, 0.8, 0.7,
    #     0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # tracelens = [10, 15, 20, 25, 30, 40, 50]
    # propsteps = [10, 15, 20, 25, 30, 40, 50]
    # runs = 50

    # # for d in discounts:
    # #     plot_tracelen_vs_static_propstep(data,
    # #         discount=d, uncetainty=uncertainties[0])

    # # plot_discount_uncertainty(data)
    # plot_power(data)
    # plot_efficiency(data)

    # data_rand = get_data(filename='/tmp/resilence_test_randomstart.pickle')
    # data = get_data(filename='/tmp/learning_30_all.pickle')

    # plot_efficiency_random_start_loc(data_rand)
    # plot_power_policy_random_startloc(data_rand)

    # plot_power_cobined(data, data_rand)
    # plot_efficiency_combined(data, data_rand)

    # combine_plots(data, data_rand)

    # data1 = get_data(filename='/tmp/mission_learning.pickle')
    # data2 = get_data(filename='/tmp/mission_learning_last.pickle')
    # data = {**data1, **data2}
    # del data1
    # del data2
    # data = get_data(filename='/tmp/mission_learning_few1.pickle')
    # tracelens = [50]
    # propsteps = [180, 200]
    # for t in tracelens:
    #     for p in propsteps:
    #         plot_power(
    #             data, tracelen=t, propsteps=p, discount=0.9,
    #             fname='mission_learning_power_')
    #         plot_efficiency(
    #             data, tracelen=t, propsteps=p, discount=0.9,
    #             fname='mission_learning_eff_')

    # data_rand = get_data(filename='/tmp/mission_test_randomstart.pickle')
    # print(data_rand)
    # plot_efficiency_random_start_loc(data_rand)
    # plot_power_policy_random_startloc(data_rand)
    data = get_data(filename='/tmp/mission_learning_few1.pickle')
    # print(data.keys())
    data_rand = get_data(filename='/tmp/mission_test_randomstart.pickle')
    # print(data_rand.keys())
    combine_plots(data, data_rand)


if __name__ =='__main__':
    main()