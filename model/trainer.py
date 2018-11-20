import torch


def train(data, model, criterion, optimizer, params):

    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states=model.get_initial_states(len(data))

    # Loop sequence length (train)
    for i in tqdm(range(0, len(data), args.bptt),desc='> Train',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen]).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1]).cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),args.clip_norm)
        optimizer.step()

    return model


def evaluate(data, model, criterion):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Eval',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen],volatile=True).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1],volatile=True).cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Log stuff
        total_loss+=loss.data.cpu().numpy()
        num_loss+=np.prod(y.size())

    return float(total_loss)/float(num_loss)
