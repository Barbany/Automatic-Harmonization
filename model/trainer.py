import torch
import torch.nn as nn

import tqdm
import numpy as np
import math


def train(dataset, model, criterion, optimizer, params, use_cuda):
    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    # Size = length of phrase - 1 ; last sample not used to predict the following one in training
    states = model.get_initial_states(params['len_seq_phrase'] - 1)

    # Loop sequence length (train)
    for data in dataset:
        chords = data[0]
        targets = data[1]
        features = data[2]

        if use_cuda:
            x = torch.autograd.Variable(chords).cuda()
            y = torch.autograd.Variable(targets).view(-1).cuda()
            cond = torch.autograd.Variable(features).cuda()
        else:
            x = torch.autograd.Variable(chords)
            y = torch.autograd.Variable(targets).view(-1)
            cond = torch.autograd.Variable(features)

        # Truncated backpropagation
        states = model.detach(
            states)  # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits, states = model.forward(x, cond, states)

        # Print chord pdf or most probable chord
        # print(logits)
        # print('Prediction is', torch.argmax(logits, dim=1).float())

        loss = criterion(logits, y.long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_norm'])
        optimizer.step()

    return model


def evaluate(dataset, model, criterion, params, use_cuda):
    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    # Size = length of phrase - 1 ; last sample not used to predict the following one in training
    states = model.get_initial_states(params['len_seq_phrase'] - 1)

    # Loop sequence length (validation)
    total_loss = 0
    num_loss = 0

    # Loop sequence length (train)
    for data in dataset:
        chords = data[0]
        targets = data[1]
        features = data[2]

        if use_cuda:
            x = torch.autograd.Variable(chords).cuda()
            y = torch.autograd.Variable(targets).view(-1).cuda()
            cond = torch.autograd.Variable(features).cuda()
        else:
            x = torch.autograd.Variable(chords)
            y = torch.autograd.Variable(targets).view(-1)
            cond = torch.autograd.Variable(features)

        # Truncated backpropagation
        states = model.detach(
            states)  # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits, states = model.forward(x, cond, states)
        loss = criterion(logits, y.long())

        # Log stuff
        total_loss += loss.data.cpu().numpy()
        num_loss += np.prod(y.size())

    return float(total_loss) / float(num_loss)
