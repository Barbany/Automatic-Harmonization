"""Argument parser and default parameters."""

import argparse

default_params = {
        'data_path': 'data/',
        'seed': 123,
        'partitions': {
            'train': 60,
            'validation': 20,
            'test': 20
            }
    }


def parse_arguments():
    """
    Set arguments from the command line when running 'run.py'. Run with option '-h' or '--help' for
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-V', '--verbose', dest='verbose', action='store_true',
        help='Provide additional details about the program. This level of detail'
             ' can be very helpful for troubleshooting problems', default=False
    )
    parser.add_argument(
        '--split_by_phrase', action='store_true',
        help='When loading the sequences of chords, split by phrases and pad those with deficient length'
             ' with respect to the maximum length. This allows to randomize partitions. If this parameter'
             ' is not provided, the sequences will be considered as continuous (resets between change of'
             ' movements and quartets) and reshaped as if they had the average sequence length.', default=False
    )
    parser.set_defaults(**default_params)

    return parser.parse_args()
