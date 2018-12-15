import json

import torch
from torch.autograd import Variable
import torch.nn.init as init

from tensorboardX import SummaryWriter

from utils.params import parse_arguments, default_params
from utils.helpers import init_random_seed, setup_results_dir, tee_stdout, make_data_loader
from model.trainer import train, evaluate
from model.rnn import RNN, Sequence

import numpy as np
import os
from tqdm import tqdm


def main(**params):
    params = dict(
        default_params,
        **params
    )
    print('*'*20 + ' Start Harmonic Sequence Predictor ' + '*'*20)

    verbose = params['verbose']

    # Check if GPU acceleration is available
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Model will be trained with GPU acceleration')

    # Set random seed to both numpy and torch (with/out CUDA)
    init_random_seed(params['seed'], use_cuda)

    # Setup results directory depending on parameters and create log file
    results_path = setup_results_dir(params)
    tee_stdout(results_path + 'log')
    print('Results path is ', results_path, ' and log is already set there')

    # Create writer for Tensorboard: Visualize plots at real time when training
    writer = SummaryWriter(log_dir=results_path + 'tboard')

    # Run Tensorboard
    # os.system('tensorboard --logdir=' + results_path + 'tboard &')

    # Get data for all partitions
    data_loader = make_data_loader(params)
    data_train = data_loader('train')
    data_validation = data_loader('validation')
    data_test = data_loader('test')

    # Check vocabulary size and feature size. In the second case don't open file but only read header
    mapping_file = params['data_path'] + 'mappings.json'
    with open(mapping_file, "r") as infile:
        mapping = json.load(infile)
    
    train_npy = params['data_path'] + 'train_' + params['split_by_phrase'] *\
     'phrase-split' + (not params['split_by_phrase']) * 'sequential' + '_clean_data.npy'
    with open(train_npy, 'rb') as f:
        np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)

    vocabulary_size = len(mapping['chord'])
    # The chord itself is also saved in the same data matrix (#Features = shape[2] - 1)
    num_features = shape[2] - 1

    if verbose:
        print('Vocabulary size is', vocabulary_size, 'and number of features is', num_features)

    # Initiate model and move it to the GPU if possible
    if use_cuda:
        model = Sequence(params['hidden_size'], vocabulary_size).cuda()
    else:
        model = Sequence(params['hidden_size'], vocabulary_size)

    # Get model with double type to avoid changing the input parameters
    model.double()

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    print('-'*22 + ' Start training ' + '-'*22)

    lr=params['learning_rate']
    best_val_loss = np.inf
    for e in tqdm(range(params['num_epochs']), desc='Epoch', ncols=100, ascii=True):
        # Train
        train_loss, model = train(data_train, model, criterion, optimizer, writer, use_cuda)

        # Validation
        val_loss = evaluate(data_validation, model, criterion, writer, use_cuda)

        if params['adaptive_lr']:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                lr /= params['anneal_factor']
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Test
        test_loss = evaluate(data_test, model, criterion, writer, use_cuda)

        # Report
        msg = 'Epoch %d: \tTrain loss=%.4f \tValidation loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(
            e+1,train_loss, val_loss,test_loss,np.exp(test_loss))
        tqdm.write(msg)
        writer.add_scalars('data/loss', {'train': train_loss,
                                         'validation': val_loss,
                                         'test': test_loss}, e)

    writer.close()


if __name__ == '__main__':
    main(**vars(parse_arguments()))
