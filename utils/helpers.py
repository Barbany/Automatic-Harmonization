"""Function used to create results folder and save a log of all prints."""

import os
import sys
import shutil
import torch
import numpy as np
from torch.utils.data import DataLoader

from model.data_loader import FolderDataset
from utils.params import default_params


tag_params = [
    'split_by_phrase', 'embedding_size', 'hidden_size'
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
    log_file = open(log_path, 'a', 1)
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


def make_data_loader(params):
    def data_loader(partition):
        dataset = FolderDataset(params['data_path'], params['split_by_phrase'],
                                partition, params['partitions'], params['verbose'])

        # Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
        # Parameters: - shuffle (If Falseon't reshuffle data at every epoch)
        #             - num_workers (Number of subprocesses to use for data loading - 0: default; increment if data loading slow)
        return DataLoader(dataset, shuffle=False, num_workers=0)
    return data_loader
