"""Function used to create results folder and save a log of all prints."""

import os
import sys
import torch
import numpy as np
from utils.params import default_params
import csv


tag_params = [
    'split_by_phrase', 'embedding', 'conditioner'
    ]


def make_tag(params):
    def to_string(value):
        if isinstance(value, bool):
            return 'T' if value else 'F'
        elif isinstance(value, list):
            return ','.join(map(to_string, value))
        else:
            return str(value)

    return '-'.join(
        key + '__' + to_string(params[key])
        for key in tag_params
        if key not in default_params or params[key] != default_params[key]
    )


def setup_results_dir(params):
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    tag = make_tag(params)
    results_path = params['results_path']
    ensure_dir_exists(results_path)
    results_path = results_path + tag + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


def tee_stdout(log_path):
    log_file = open(log_path, 'w', 1)
    stdout = sys.stdout

    class Tee:
        @staticmethod
        def write(string):
            log_file.write(string)
            stdout.write(string)

        @staticmethod
        def flush():
            log_file.flush()
            stdout.flush()

    sys.stdout = Tee()


def init_random_seed(seed, cuda):
    print('Seed: ', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def save_test_losses_to_tsv(params):
    """
    This function computes the median, 1st and 3rd quantile, maximum and minimum value for the test losses ef each model
    and save the date into a .csv in order to draw boxplots.
    """
    path = params['results_path'] + params['test_losses_path']
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    with open(path + 'boxplots_data.csv', 'w') as csvfile:
        boxplots = csv.writer(csvfile, delimiter=',')
        boxplots.writerow(['Name', 'First Quantile', 'Median', 'Third Quantile', 'Maximum', 'Minimum'])

    for file in files:
        if file.endswith(".npy"):
            loss_array = np.load(path + file)
            first_quant = np.quantile(loss_array, 0.25)
            median = np.quantile(loss_array, 0.5)
            third_quant = np.quantile(loss_array, 0.75)
            max_value = np.amax(loss_array)
            min_value = np.amin(loss_array)
            with open(path + 'boxplots_data.csv', 'a') as csvfile:
                boxplots = csv.writer(csvfile, delimiter=',')
                boxplots.writerow([file[:-4], first_quant, median, third_quant, max_value, min_value])