#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os
import sys
import re
from datetime import datetime

plt.rcParams.update({'figure.max_open_warning': 100})


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


def get_time(log_entry):
    timestamp_pattern = r"\[(\d{4}/\d{2}/\d{2}_\d{2}:\d{2}:\d{2}\.\d{3})\]"
    m = re.search(timestamp_pattern, log_entry)
    if m is None:
        raise ValueError(f"Cannot parse timestamp from log entry: {log_entry}")
    time_str = m.group(1)
    return datetime.strptime(time_str, "%Y/%m/%d_%H:%M:%S.%f")


def safe_positive_int(value, fallback=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def analysis(training_dir, path, iter: int = -1, all: bool = False, name: bool = False):
    path = os.path.join(training_dir, path)
    if not os.path.isdir(path):
        os.mkdir(path)
    analysis_(training_dir, path, iter, all, name)


def get_myDict(lines, iter):
    set_learner_training_display_step = True
    set_learner_training_step = True
    learner_training_display_step = 0
    learner_training_step = 0
    myDict = {"[Iteration]": [], "Time": [], "OP Time": [], "SP Time": []}

    # parse log
    for index in range(2):
        counter = 1
        Start_time = None
        OP_start_time = None

        for line in lines[index]:
            if not line:
                continue

            if iter != -1 and counter >= iter:
                break

            ret3 = re.findall(r'(loss|accuracy)_(\S+)(_\d)?:\s(-?\d+\.?\d*(?:[Ee]-?\d+)?)', line)
            if ret3:
                key = ret3[0][0] + "_" + ret3[0][1] + ret3[0][2]
                if key not in myDict:
                    myDict[key] = []
                myDict[key].append(float(ret3[0][3]))
                continue

            ret1 = re.findall(r'(\[SelfPlay\s(?:Min\.|Max\.|Avg\.)\s(?!Data Lengths\])(.*?)\])\s(-?\d+\.\d+|\d+)', line)
            if ret1:
                key = ret1[0][0]
                if key not in myDict:
                    myDict[key] = []
                myDict[key].append(float(ret1[0][2]))
                if re.findall(r'(\[SelfPlay Avg. Game Returns\])', line):
                    myDict["[Iteration]"].append(counter)
                continue

            ret2 = re.findall(r'((\[Iteration\])\s={5}(\d+)={5})', line)
            if ret2:
                counter += 1

            if index == 0 and re.findall(r'(Optimization_Done)\s(\d+)', line):
                counter += 1

            if index == 0:  # op.log
                ret4 = re.findall(r'(nn\sstep)\s(\d+),', line)
                if ret4 and set_learner_training_display_step:
                    set_learner_training_display_step = False
                    learner_training_display_step = int(ret4[0][1])

                ret5 = re.findall(r'(Optimization_Done)\s(\d+)', line)
                if ret5 and set_learner_training_step:
                    learner_training_step = int(ret5[0][1])
                    set_learner_training_step = False

            if index == 1:  # Training.log
                ret1 = re.findall(r'((\[.*\])\s\[Iteration\]\s={5}(\d+)={5})', line)
                if ret1:
                    Start_time = get_time(ret1[0][1])

                ret2 = re.findall(r'((\[.*\])\s\[Optimization\]\sStart.)', line)
                if ret2:
                    OP_start_time = get_time(ret2[0][1])

                ret3 = re.findall(r'((\[.*\])\s\[Optimization\]\sFinished.)', line)
                if ret3 and Start_time is not None and OP_start_time is not None:
                    Finished_time = get_time(ret3[0][1])
                    myDict["Time"].append((Finished_time - Start_time).total_seconds())
                    myDict["OP Time"].append((Finished_time - OP_start_time).total_seconds())
                    myDict["SP Time"].append((OP_start_time - Start_time).total_seconds())

    return myDict, learner_training_display_step, learner_training_step


def graph_print(tmp, iter):
    lines = []
    op_lines = []
    myDict = {}

    for item in tmp:
        with open(os.path.join(item, "Training.log"), 'r') as log:
            with open(os.path.join(item, "op.log"), 'r') as op_log:
                lines.append(log.readlines())
                op_lines.append(op_log.readlines())

    nn_step = []
    for index in range(len(op_lines)):
        set_learner_training_step = True
        for line in op_lines[index]:
            ret5 = re.findall(r'(Optimization_Done)\s(\d+)', line)
            if ret5 and set_learner_training_step:
                nn_step.append(int(ret5[0][1]))
                set_learner_training_step = False

        if set_learner_training_step:
            nn_step.append(1)

    for index in range(len(lines)):
        counter = 0
        for line in lines[index]:
            if iter != -1 and counter >= iter:
                break

            ret2 = re.findall(r'((\[Iteration\])\s={5}(\d+)={5})', line)
            if ret2:
                counter = int(ret2[0][2])

            ret1 = re.findall(r'(\[SelfPlay Avg. Game Returns\])\s(-?\d+\.\d+|\d+)', line)
            if ret1:
                key = f"[SelfPlay Avg. Game Returns] {index}"
                if key not in myDict:
                    myDict[key] = []
                myDict[key].append(float(ret1[0][1]))

                iter_key = f"[Iteration] {index}"
                if iter_key not in myDict:
                    myDict[iter_key] = []
                myDict[iter_key].append(counter)

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(25, 20))
    bool_print = False

    for index in range(len(tmp)):
        width = 4
        x = [x * safe_positive_int(nn_step[index], 1) for x in myDict.get(f"[Iteration] {index}", [])]
        y = myDict.get(f"[SelfPlay Avg. Game Returns] {index}", [])
        if x and y:
            bool_print = True
            plt.plot(x, y, label=tmp[index], linewidth=width)

    if bool_print:
        plt.title('[SelfPlay Avg. Game Returns] graph', fontsize=30)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
        plt.xlabel('nn steps', fontsize=30)
        plt.ylabel('Return', fontsize=30)
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig("compare_graph")
        eprint("compare_graph")


def format_y_axis_labels(value, pos):
    if value >= 1000:
        value /= 1000
        return f'{value:.0f}k'
    return f'{value:.2f}'


def set_iteration_xlim(ax_bottom, ax_top, learner_training_step):
    step = safe_positive_int(learner_training_step, 0)
    if step <= 0:
        return

    x0, x1 = ax_bottom.get_xlim()
    if not all(map(lambda v: v == v and v not in (float("inf"), float("-inf")), [x0, x1])):
        return

    ax_top.set_xlim([x0 / step, x1 / step])


def analysis_(dir, path, iter, all: bool = False, name: bool = False):
    with open(os.path.join(dir, "op.log"), 'r') as op_log:
        with open(os.path.join(dir, "Training.log"), 'r') as Training_log:
            lines = [op_log.readlines(), Training_log.readlines()]

    myDict, learner_training_display_step, learner_training_step = get_myDict(lines, iter)

    sp_items = list(set([
        re.search(r'\[SelfPlay (?:Min\.|Max\.|Avg\.) (.*?)\]', key).group(1).replace("Game ", "")
        for key in myDict
        if re.match(r'^\[SelfPlay (?:Min\.|Max\.|Avg\.) .*?\]', key)
    ]))
    op_items = list(set([
        re.sub(r"_\d+$", "", key)
        for key in myDict
        if re.match(r'^(loss|accuracy)_', key)
    ]))
    Fig_list = list(set(sp_items + op_items + ["Time"]))

    counter_subplot = len(Fig_list)
    if counter_subplot == 0 or len(myDict["Time"]) == 0:
        return

    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1, counter_subplot, figsize=(25 * counter_subplot, 20))

    if counter_subplot == 1:
        axs = [axs]

    counter_fig = 0
    original_dir = dir

    for item in sorted(Fig_list):
        fig_one = plt.figure(figsize=(25, 20))
        ax1 = fig_one.add_subplot(111)
        ax2 = ax1.twiny()

        width = 4
        legend_fontsize = 30
        bool_print = False

        for key in sorted(myDict.keys()):
            if re.search(r'SelfPlay (?:Min\.|Max\.|Avg\.) ' + item, key.replace("Game ", "")):
                bool_print = True
                step_interval = safe_positive_int(learner_training_step, 1)
                legend_fontsize = min(legend_fontsize, 30 if len(key) <= 30 else 20)
                linecolor = 'red' if "Avg." in key else None

                x = [x * step_interval for x in myDict["[Iteration]"][-len(myDict[key]):]]
                y = myDict[key]

                ax1.plot(x, y, label=f'{key}', linewidth=width, color=linecolor)
                axs[counter_fig].plot(x, y, label=f'{key}', linewidth=width, color=linecolor)
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_axis_labels))

            elif "SelfPlay" not in key and item in key:
                bool_print = True
                step_interval = safe_positive_int(
                    learner_training_step if "Time" in item else learner_training_display_step,
                    1
                )
                legend_fontsize = min(legend_fontsize, 30 if len(key) <= 30 else 20)

                x = [(x + 1) * step_interval for x in range(len(myDict[key]))]
                y = myDict[key]

                ax1.plot(x, y, label=f'{key}', linewidth=width)
                axs[counter_fig].plot(x, y, label=f'{key}', linewidth=width)

        if bool_print:
            plt.title(f'{item} of {original_dir}', fontsize=30)
            axs[counter_fig].set_title(f'{item} of {original_dir}')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=legend_fontsize)
            axs[counter_fig].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=legend_fontsize)

            ax1.set_xlabel('nn steps', fontsize=30)
            ax2.set_xlabel('iterations', fontsize=30)

            axs_twiny = axs[counter_fig].twiny()
            axs_twiny.set_xlabel('iterations', fontsize=30)

            set_iteration_xlim(ax1, ax2, learner_training_step)
            set_iteration_xlim(ax1, axs_twiny, learner_training_step)

            ax1.set_ylabel(f'{item}', fontsize=30)
            axs[counter_fig].set(xlabel='nn steps', ylabel=f'{item}')
            ax1.grid()
            axs[counter_fig].grid()
            plt.tight_layout()

            out_dir = os.path.dirname(original_dir + "/")
            fig_file = os.path.join(path, f'{out_dir.split("/")[-1]}_{re.sub(r"[^A-Za-z0-9]+", "_", item).strip("_")}.png')
            plt.savefig(fig_file)
            if name:
                eprint(fig_file)

            plt.cla()
            counter_fig += 1

    if all:
        fig.tight_layout()
        all_file = os.path.join(path, f'{str(original_dir.split("/")[-1])}.png')
        fig.savefig(all_file)
        eprint(all_file)

    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', dest='in_dir', default='', help='dir to anaylisis')
    parser.add_argument('-out_dir', dest='out_dir', type=str, default="analysis_", help='output directory (default: analysis_{iter})')
    parser.add_argument('-iter', dest='iter', default=-1, type=int, help='iteration to anaylisis')
    parser.add_argument('-compare', dest='compare', default='', type=str, help='compare many version of same game, use commas as separators. ex: -compare a,b')
    parser.add_argument('--all', dest='all', default=False, action="store_true", help='all analysis in a graph, default false')
    args = parser.parse_args()

    if args.iter != -1:
        out_dir = args.out_dir + str(args.iter)
    else:
        out_dir = args.out_dir

    compare = args.compare
    tmp = compare.split(',')

    if args.in_dir:
        dir = args.in_dir
        if dir and os.path.isdir(dir):
            name = True
            path = os.path.join(dir, f'{out_dir}')
            if not os.path.isdir(path):
                os.mkdir(path)
            analysis_(dir, path, args.iter, args.all, name)
        else:
            eprint(f'"{dir}" does not exist!')
            exit(1)
    elif tmp != ['']:
        for item in tmp:
            if not os.path.isdir(item):
                eprint(f'"{item}" does not exist!')
                exit(1)
        graph_print(tmp, args.iter)
    else:
        parser.print_help()
        exit(1)