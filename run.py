import json

import torch

from tensorboardX import SummaryWriter

from utils.params import parse_arguments, default_params
from utils.helpers import init_random_seed, setup_results_dir, tee_stdout

import numpy as np

from tqdm import tqdm

from model.rnn import Sequence
from model.generate_and_load_data import generate_data, load_data


def main(**params):
    params = dict(
        default_params,
        **params
    )
    print('*' * 20 + ' Start Harmonic Sequence Predictor ' + '*' * 20)

    verbose = params['verbose']
    conditioner = params['conditioner']
    # Force to have embedding if training with features
    if conditioner:
        embedding = True
    else:
        embedding = params['embedding']

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

    # Generate data files if not created
    generate_data(params['data_path'], params['split_by_phrase'], params['len_seq_phrase'],
                                params['len_phrase'], params['partitions'], params['verbose'])

    # Load data and make training set
    input, target, features = load_data(params['data_path'], 'train', params['split_by_phrase'])
    val_input, val_target, val_features = load_data(params['data_path'], 'validation', params['split_by_phrase'])
    test_input, test_target, test_features = load_data(params['data_path'], 'test', params['split_by_phrase'])

    # Check vocabulary size and feature size. In the second case don't open file but only read header
    mapping_file = params['data_path'] + 'mappings.json'
    with open(mapping_file, "r") as infile:
        mapping = json.load(infile)

    vocabulary_size = len(mapping['chord'])
    assert features.shape[2] == val_features.shape[2] and features.shape[2] == test_features.shape[2]

    if conditioner:
        num_features = features.shape[2]
    else:
        num_features = None

    if verbose:
        print('*' * 22 + ' TRAINING DETAILS ' + '*' * 22)
        print('Vocabulary size is', vocabulary_size, ('and number of features is ' + str(num_features)) * conditioner)
        if embedding:
            print('Using embedding with size', params['embedding_size'])
        if params['split_by_phrase']:
            print('Using approach of split by phrase (randomized) with phrases with a length of', params['len_phrase'])
        else:
            print('Using approach of sequential split with phrases with a length of', params['len_seq_phrase'])

    # Build the model and move it to the GOU is possible
    if use_cuda:
        seq = Sequence(vocabulary_size=vocabulary_size, hidden_size=params['hidden_size'], embedding=embedding,
                       embed_size=params['embedding_size'], cond_size=num_features,
                       input_rnn_size=params['input_rnn_size']).cuda()
    else:
        seq = Sequence(vocabulary_size=vocabulary_size, hidden_size=params['hidden_size'], embedding=embedding,
                       embed_size=params['embedding_size'], cond_size=num_features,
                       input_rnn_size=params['input_rnn_size'])

    seq.double()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(seq.parameters(), lr=params['learning_rate'])

    if params['lr_controller']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params['Tm'], eta_min=0)

    # Begin to train
    print('\n' + '-' * 22 + ' Start training ' + '-' * 22)

    best_loss = np.inf

    for i in tqdm(range(params['num_epochs']), desc='Epoch', ncols=100, ascii=True):

        if params['lr_controller']:
            scheduler.step()

        optimizer.zero_grad()
        out = seq(input, features)
        train_loss = criterion(out.view(-1, out.size(2)), target.contiguous().view(-1).long())
        train_loss.backward()

        if params['gradient_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(seq.parameters(), params['gradient_clip'])

        optimizer.step()

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = seq(val_input, val_features)
            y_pred = pred.contiguous().view(-1, pred.size(2))
            y = val_target.contiguous().view(-1).long()
            val_loss = criterion(y_pred, y)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(seq.state_dict(), results_path + '/best_model.torch')

        # Tensorboard
        writer.add_scalars('data/loss', {'train': train_loss,
                                         'validation': val_loss}, i)

        # Report
        msg = 'Epoch %d: \tTrain loss=%.4f \tValidation loss=%.4f' % (i + 1, train_loss, val_loss)
        tqdm.write(msg)

    if embedding:
        seq.show_embedding(mapping, writer)
    writer.close()

    n_test = test_target.shape[0]
    # we divide the test dataset into 10 partitions to get more representative results
    k_test = 10
    n_partition_test = int(np.round(n_test/k_test))

    test_losses = []

    for k in range(k_test):
        if k != k_test-1:
            test_input_part = test_input[k*n_partition_test:(k+1)*n_partition_test]
            test_features_part = test_features[k*n_partition_test:(k+1)*n_partition_test]
            test_target_part = test_target[k*n_partition_test:(k+1)*n_partition_test]
        else:
            test_input_part = test_input[k*n_partition_test:]
            test_features_part = test_features[k*n_partition_test:]
            test_target_part = test_target[k*n_partition_test:]

        with torch.no_grad():
            pred = seq(test_input_part, test_features_part)
            y_pred = pred.contiguous().view(-1, pred.size(2))
            y = test_target_part.contiguous().view(-1).long()
            test_loss = criterion(y_pred, y)
            test_losses.append(test_loss)

    mean_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    print('Test loss = %.4f +- %.4f' % (mean_test_loss, std_test_loss))


if __name__ == '__main__':
    main(**vars(parse_arguments()))
