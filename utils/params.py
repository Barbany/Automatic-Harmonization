"""Argument parser and default parameters."""

import argparse

default_params = {
    'data_path': 'data/',
    'results_path': 'results/',
    'test_losses_path': 'test_losses_arrays/',
    'seed': 123,
    'partitions': {
        'train': 80,
        'validation': 10,
        'test': 10
        },
    'embedding_size': 7,    # Hypothesis: 7 (number of notes) or 12 (tones in equal temperament)
    'hidden_size': 51,
    'learning_rate': 0.01,
    'num_epochs': 500,
    'input_rnn_size': 20,   # size of the output of the FCL (input of the RNN when training with features)
    'Tm': 50,   # maximum number of iterations (of the learning rate scheduler)
    'lr_controller': True,
    'gradient_clip': 0.5,
    'len_seq_phrase': 30,   # length of the phrases in the approach of sequential split
    'len_phrase': 20    # length of phrases in the approach with randomized phrases (the complete phrases are too long)
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
             ' can be very helpful for troubleshooting problems.', default=False
    )
    parser.add_argument(
        '--split_by_phrase', action='store_true',
        help='When loading the sequences of chords, split by phrases and pad those with deficient length'
             ' with respect to the maximum length. This allows to randomize partitions. If this parameter'
             ' is not provided, the sequences will be considered as continuous (resets between change of'
             ' movements and quartets) and reshaped as if they had the average sequence length.', default=False
    )
    parser.add_argument(
        '--embedding', action='store_false',
        help='When training the model, do not embed the chords. Instead, use directly the chords.', default=True
    )
    parser.add_argument(
        '--conditioner', action='store_false',
        help='When training the model, use features in addition to the chords. If not, use only the chords.'
             'When using features, embedding of chords is always used.', default=True
    )
    parser.set_defaults(**default_params)

    return parser.parse_args()


def parse_arguments_prediction():
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
    parser.add_argument(
        '--embedding', action='store_false',
        help='When training the model, do not embed the chords. Instead, use directly the chords.', default=True
    )
    parser.add_argument(
        '--conditioner', action='store_false',
        help='When training the model, use features in addition to the chords. If not, use only the chords.'
             'When using features, embedding of chords is always used.', default=True
    )
    parser.add_argument(
        '--input_file',
        help='Input file with the first chords and the features. If not provided, a prompt will ask for the desired length'
             ' and default features will be used.',
             default='example_input.tsv'
             )
    parser.add_argument(
        '--output_file',
        help='Output file to store the predicted chords.',
             default='output.tsv'
             )
    parser.set_defaults(**default_params)

    return parser.parse_args()
