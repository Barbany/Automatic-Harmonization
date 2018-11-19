import torch
from torch.autograd import Variable
import torch.nn.init as init

from tensorboardX import SummaryWriter

from utils.params import parse_arguments, default_params
from utils.helpers import init_random_seed, setup_results_dir, tee_stdout
from model.trainer import forward

import numpy as np
import subprocess


def main(**params):
    params = dict(
        default_params,
        **params
    )
    print('-'*20 + ' Start Harmonic Sequence Predictor ' + '*'*20)

    # Check if GPU acceleration is available
    use_cuda = torch.cuda.is_available()

    # Set random seed to both numpy and torch (with/out CUDA)
    init_random_seed(params['seed'], use_cuda)

    # Setup results directory depending on parameters and create log file
    results_path = setup_results_dir(params)
    tee_stdout(results_path + 'log')

    # Create writer for Tensorboard: Visualize plots at real time when training
    writer = SummaryWriter(log_dir=results_path + 'tboard')

    # Run Tensorboard
    subprocess.run('tensorboard --logdir ' + results_path + 'tboard &')

    dtype = torch.FloatTensor
    input_size, hidden_size, output_size = 7, 6, 1
    epochs = 300
    seq_length = 20
    lr = 0.1

    data_time_steps = np.linspace(2, 10, seq_length + 1)
    data = np.sin(data_time_steps)
    data.resize((seq_length + 1, 1))

    x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
    y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)

    w1 = torch.FloatTensor(input_size, hidden_size).type(dtype)
    init.normal_(w1, 0.0, 0.4)
    w1 = Variable(w1, requires_grad=True)
    w2 = torch.FloatTensor(hidden_size, output_size).type(dtype)
    init.normal_(w2, 0.0, 0.3)
    w2 = Variable(w2, requires_grad=True)

    for i in range(epochs):
        context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
        total_loss = 0
        for j in range(x.size(0)):
            _input = x[j:(j + 1)]
            target = y[j:(j + 1)]
            (pred, context_state) = forward(_input, context_state, w1, w2)
            loss = (pred - target).pow(2).sum() / 2
            total_loss += loss
            loss.backward()
            w1.data -= lr * w1.grad.data
            w2.data -= lr * w2.grad.data
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            context_state = Variable(context_state.data)
        writer.add_scalar('data/loss', total_loss, i)
        if i % 10 == 0:
            print("Epoch: {} loss {}".format(i, total_loss.item()))

    context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
    predictions = []

    for i in range(x.size(0)):
        _input = x[i:i + 1]
        (pred, context_state) = forward(_input, context_state, w1, w2)
        context_state = context_state
        predictions.append(pred.data.numpy().ravel()[0])

    writer.close()


if __name__ == '__main__':
    main(**vars(parse_arguments()))
