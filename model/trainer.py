import torch
import tqdm


def train(dataset, model, criterion, optimizer, params, use_cuda):

    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states = model.get_initial_states(len(data))

    # Loop sequence length (train)
    for data in dataset:
        inputs = data[0]
        targets = data[1]

        if use_cuda:
            x = torch.autograd.Variable(inputs).cuda()
            y = torch.autograd.Variable(targets).cuda()
        else:
            x = torch.autograd.Variable(inputs)
            y = torch.autograd.Variable(targets)
        
        # Truncated backpropagation
        states = model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits, states = model.forward(x, states)
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), params['clip_norm'])
        optimizer.step()

    return model


def evaluate(data, model, criterion, use_cuda):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0

    # Loop sequence length (train)
    for data in dataset:
        inputs = data[0]
        targets = data[1]

        if use_cuda:
            x = torch.autograd.Variable(inputs).cuda()
            y = torch.autograd.Variable(targets).cuda()
        else:
            x = torch.autograd.Variable(inputs)
            y = torch.autograd.Variable(targets)
        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits, states = model.forward(x,states)
        loss = criterion(logits, y)

        # Log stuff
        total_loss += loss.data.cpu().numpy()
        num_loss += np.prod(y.size())

    return float(total_loss)/float(num_loss)
