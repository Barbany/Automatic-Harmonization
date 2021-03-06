import json
import os

import numpy as np
import pandas as pd
import torch

from model.rnn import Sequence
from utils.helpers import init_random_seed, setup_results_dir
from utils.params import parse_arguments_prediction, default_params


def main(**params):
    params = dict(
        default_params,
        **params
    )

    results_path = setup_results_dir(params)
    if not os.path.exists(results_path + 'samples/'):
        os.makedirs(results_path + 'samples/')

    # Check vocabulary size and feature size. In the second case don't open file but only read header
    mapping_file = params['data_path'] + 'mappings.json'
    with open(mapping_file, "r") as infile:
        mapping = json.load(infile)

    
    df = pd.read_csv(params['input_file'], sep='\t')

    for col in df.columns:
        if col in mapping:
            df[col] = np.vectorize(mapping[col].get)(df[col])

    features = np.asarray(df.values, dtype=float)

    chords = features[:, 0]
    features = features[:, 1:]
    
    split_by_phrase = params['split_by_phrase']
    mean_features = np.load(params['data_path'] + split_by_phrase * 'phrase-split' +
                         (not split_by_phrase) * 'sequential' + '_mean_features.npy')
    std_features = np.load(params['data_path'] + split_by_phrase * 'phrase-split' +
                         (not split_by_phrase) * 'sequential' + '_std_features.npy')               
    features = (features - mean_features) / std_features

    # Keep non-nan values (initial chords) and define the future (chords to be predicted given the length of features)
    chords = chords[~np.isnan(chords)]
    future = len(features) - len(chords)

    print('*' * 20 + ' Start Harmonic Sequence Predictor ' + '*' * 20)
    print('Given your', len(chords), 'initial chords, our system will predict a sequence of length', future, 
        'additional chords conditioned on your features')

    conditioner = params['conditioner']

    if conditioner:
        embedding = True
    else:
        embedding = params['embedding']

    # Check if GPU acceleration is available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Sequence will be predicted with GPU acceleration')

    # Set random seed to both numpy and torch (with/out CUDA)
    init_random_seed(params['seed'], use_cuda)

    vocabulary_size = len(mapping['chord'])
    
    if conditioner:
        num_features = features.shape[1]
    else:
        num_features = None

    # Build the model and move it to the GOU is possible
    if use_cuda:
        seq = Sequence(vocabulary_size=vocabulary_size, hidden_size=params['hidden_size'], embedding=embedding,
                       embed_size=params['embedding_size'], cond_size=num_features,
                       input_rnn_size=params['input_rnn_size']).cuda()
    else:
        seq = Sequence(vocabulary_size=vocabulary_size, hidden_size=params['hidden_size'], embedding=embedding,
                       embed_size=params['embedding_size'], cond_size=num_features,
                       input_rnn_size=params['input_rnn_size'])
    seq.load_state_dict(torch.load(results_path + 'best_model.torch'))
    seq.double()

    # Convert chords and inputs to torch tensors
    chords = torch.from_numpy(chords).view(1, -1)
    features = torch.from_numpy(features).view(1, -1, num_features)

    with torch.no_grad():
        pred = seq(chords, features, future=future)
        pred = torch.argmax(pred, dim=2).view(-1).double()
    
    # Translate predicted chords into roman numerals and put them in a TSV output file
    reverse_mapping = dict((v,k) for k,v in mapping['chord'].items())
    pred = np.vectorize(reverse_mapping.get)(pred)

    df = pd.read_csv(params['input_file'], sep='\t')
    df['chord'] = df['chord'].fillna(pd.Series(pred))

    out_file = results_path + 'samples/' + params['output_file']
    df.to_csv(out_file, sep='\t', index=False)
    print('The full chord sequence has been saved in', out_file)


if __name__ == '__main__':
    main(**vars(parse_arguments_prediction()))
