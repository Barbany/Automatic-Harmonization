import json

import torch

from tensorboardX import SummaryWriter

from utils.params import parse_arguments, default_params
from utils.helpers import init_random_seed, setup_results_dir, tee_stdout

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

    # Load data and make training set (we do not use the features for the moment - use only the chords)
    input, target, features = load_data(params['data_path'], 'train', params['split_by_phrase'])
    val_input, val_target, val_features = load_data(params['data_path'], 'validation', params['split_by_phrase'])
    test_input, test_target, test_features = load_data(params['data_path'], 'test', params['split_by_phrase'])

    # Check vocabulary size and feature size. In the second case don't open file but only read header
    mapping_file = params['data_path'] + 'mappings.json'
    with open(mapping_file, "r") as infile:
        mapping = json.load(infile)

    vocabulary_size = len(mapping['chord'])
    assert features.shape[2] == val_features.shape[2] and features.shape[2] == test_features.shape[2]
    num_features = features.shape[2]

    if verbose:
        print('*' * 22 + ' TRAINING DETAILS ' + '*' * 22)
        print('Vocabulary size is', vocabulary_size, 'and number of features is', num_features)
        if params['embedding']:
            print('Using embedding with size', params['embedding_size'])
        if params['split_by_phrase']:
            print('Using approach of split by phrase (randomized) with phrases with a length of', params['len_phrase'])
        else:
            print('Using approach of sequential split with phrases with a length of', params['len_seq_phrase'])

    # Build the model and move it to the GOU is possible
    if use_cuda:
        seq = Sequence(vocabulary_size, params['hidden_size'], params['embedding'], params['embedding_size']).cuda()
    else:
        seq = Sequence(vocabulary_size, params['hidden_size'], params['embedding'], params['embedding_size'])

    seq.double()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(seq.parameters(), lr=params['learning_rate'])

    # Begin to train
    print('\n' + '-' * 22 + ' Start training ' + '-' * 22)

    for i in tqdm(range(params['num_epochs']), desc='Epoch', ncols=100, ascii=True):

        optimizer.zero_grad()
        out = seq(input)
        train_loss = criterion(out.view(-1, out.size(2)), target.contiguous().view(-1).long())
        train_loss.backward()

        optimizer.step()

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = seq(val_input)
            y_pred = pred.contiguous().view(-1, pred.size(2))
            y = val_target.contiguous().view(-1).long()
            val_loss = criterion(y_pred, y)

        # Tensorboard
        writer.add_scalars('data/loss', {'train': train_loss,
                                         'validation': val_loss}, i)

        # Report
        msg = 'Epoch %d: \tTrain loss=%.4f \tValidation loss=%.4f' % (i + 1, train_loss, val_loss)
        tqdm.write(msg)

    writer.close()

    with torch.no_grad():
        pred = seq(test_input)
        y_pred = pred.contiguous().view(-1, pred.size(2))
        y = test_target.contiguous().view(-1).long()
        loss = criterion(y_pred, y)
        print('Test loss =', loss.item())


if __name__ == '__main__':
    main(**vars(parse_arguments()))
