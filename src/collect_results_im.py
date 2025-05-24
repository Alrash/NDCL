# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

import datasets
import algorithms
from lib import misc, reporting
import model_selection
from lib.query import Q
import warnings

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

# mode: 0 total; 1 total + target; 2 total + target + source
def print_results_tables(records, selection_method, latex, mode: int = 0, show_mean: bool = True):
    """Given all records, print a results table for each dataset."""
    # grouped_records = reporting.get_grouped_records(records).map(lambda group:
    #     { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    # ).filter(lambda g: g["sweep_acc"] is not None)
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        {**group, **selection_method.sweep_acc(group['records'])})
    grouped_keys = grouped_records[0].keys()
    for key in grouped_keys:
        if key.split('_')[-1] in ['many', 'median', 'few', 'zero']:
            grouped_records = grouped_records.filter(lambda g: g[key] is not None)

    # obtain labels
    if mode == 0:
        labels = ['test_acc']
    elif mode == 1:
        labels = ['test_' + item for item in ['acc', 'many', 'median', 'few', 'zero'] if 'test_' + item in grouped_keys]
    elif mode == 2:
        labels = []
        for domain in ['test_', 'source_']:
            for stat in ['acc', 'many', 'median', 'few', 'zero']:
                if domain + stat in grouped_keys:
                    labels.append(domain + stat)
    else:
        raise NotImplementedError('>_<')

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        # table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        num_envs = datasets.num_environments(dataset)
        if show_mean:
            table = [[None for _ in range((num_envs + 1) * len(labels))] for _ in alg_names]
        else:
            table = [[None for _ in range(num_envs * len(labels))] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = [[] for _ in range(len(labels))]
            for j, test_env in enumerate(test_envs):
                for ind in range(len(labels)):
                    trial = (grouped_records.filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)).select(labels[ind]))
                    mean, err, table[i][j * len(labels) + ind] = format_mean(trial, latex)
                    means[ind].append(mean)
            if show_mean:
                for ind in range(len(labels)):
                    if None in means[ind]:
                        table[i][num_envs * len(labels) + ind] = "X"
                    else:
                        table[i][num_envs * len(labels) + ind] = "{:.1f}".format(sum(means[ind]) / len(means[ind]))

        if show_mean:
            cols = [envs + ' ' + l.replace('test_', '').capitalize()
                    for envs in [*datasets.get_dataset_class(dataset).ENVIRONMENTS, 'Avg'] for l in labels]
        else:
            cols = [envs + ' ' + l.replace('test_', '').capitalize()
                    for envs in [*datasets.get_dataset_class(dataset).ENVIRONMENTS] for l in labels]
        # col_labels = [
        #     "Algorithm",
        #     *datasets.get_dataset_class(dataset).ENVIRONMENTS,
        #     "Avg"
        # ]
        col_labels = ['Algorithm', *cols]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    num_datasets = len(dataset_names)
    table = [[None for _ in range((num_datasets + 1) * len(labels))] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = [[] for _ in range(len(labels))]
        for j, dataset in enumerate(dataset_names):
            for ind in range(len(labels)):
                trial_averages = (grouped_records
                    .filter_equals("algorithm, dataset", (algorithm, dataset))
                    .group("trial_seed")
                    .map(lambda trial_seed, group:
                        group.select(labels[ind]).mean()
                    )
                )
                mean, err, table[i][j * len(labels) + ind] = format_mean(trial_averages, latex)
                means[ind].append(mean)
        for ind in range(len(labels)):
            if None in means[ind]:
                table[i][num_datasets * len(labels) + ind] = "X"
            else:
                table[i][num_datasets * len(labels) + ind] = "{:.1f}".format(sum(means[ind]) / len(means[ind]))

    # col_labels = ["Algorithm", *dataset_names, "Avg"]
    cols = [envs + ' ' + l.replace('test_', '').capitalize()
            for envs in [*dataset_names, 'Avg'] for l in labels]
    col_labels = ["Algorithm", *cols]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument('--mode', type=int, default=1)
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethodImbalanced,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex, args.mode)

    if args.latex:
        print("\\end{document}")
