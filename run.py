import json

import torch
from torch.autograd import Variable
import torch.nn.init as init

from tensorboardX import SummaryWriter

from utils.params import parse_arguments, default_params
from utils.helpers import init_random_seed, setup_results_dir, tee_stdout, make_data_loader
from model.trainer import train, evaluate
from model.rnn import RNN

import numpy as np
import os
from tqdm import tqdm


def main(**params):
    params = dict(
        default_params,
        **params
    )
    print('*'*20 + ' Start Harmonic Sequence Predictor ' + '*'*20)

    # Check if GPU acceleration is available
    use_cuda = torch.cuda.is_available()

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

    mapping_file = params['data_path'] + 'mappings.json'
    with open(mapping_file, "r") as infile:
        mapping = json.load(infile)

    vocabulary_size = len(mapping['chord'])

    # Initiate model and move it to the GPU if possible
    if use_cuda:
        model = RNN(vocabulary_size, params['embedding_size'], params['hidden_size'], use_cuda).cuda()
    else:
        model = RNN(vocabulary_size, params['embedding_size'], params['hidden_size'], use_cuda)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss(size_average=False)

    # Define optimizer
    optimizer=torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

    print('-'*20 + ' Start training ' + '-'*20)

    lr=params['learning_rate']
    best_val_loss = np.inf
    for e in tqdm(range(params['num_epochs']), desc='Epoch', ncols=100, ascii=True):
        # Train
        model = train(data_train, model, criterion, optimizer, params, use_cuda)
        train_loss = evaluate(data_train, model, criterion, params, use_cuda)

        # Validation
        val_loss = evaluate(data_validation, model, criterion, params, use_cuda)

        # Anneal learning rate
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            lr /= params['anneal_factor']
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Test
        test_loss = evaluate(data_test, model, criterion, params, use_cuda)

        # Report
        msg = 'Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,val_loss,test_loss,np.exp(test_loss))
        tqdm.write(msg)
        writer.add_scalars('data/loss', {'train': train_loss,
                                         'validation': val_loss,
                                         'test': test_loss}, e)

    writer.close()


if __name__ == '__main__':
    main(**vars(parse_arguments()))
