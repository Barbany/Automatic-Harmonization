import torch

def train(dataset, model, criterion, optimizer, writer, use_cuda):
    data = dataset
    chords = data[0]
    targets = data[1]
    features = data[2]
    if use_cuda:
        chords = chords.cuda()
        targets = targets.cuda()
        features = features.cuda()

    optimizer.zero_grad()
    out = model(chords, writer=writer)
    loss = criterion(out.view(-1, out.size(2)), targets.contiguous().view(-1).long())
    loss.backward()
    optimizer.step()
    return loss.item(), model


def evaluate(dataset, model, criterion, writer, use_cuda):
    data = dataset
    chords = data[0]
    targets = data[1]
    features = data[2]
    if use_cuda:
        chords = chords.cuda()
        targets = targets.cuda()
        features = features.cuda()
    with torch.no_grad():
        chords_pred = model(chords, writer=writer)
        loss = criterion(chords_pred.view(-1, chords_pred.size(2)), targets.contiguous().view(-1).long())
    return loss.item()


def predict(dataset, model, use_cuda):
    for data in dataset:
        chords = data[0]
        features = data[1]
        if use_cuda:
            chords = chords.cuda()
            features = features.cuda()
        
        future = len(features)

        chords_pred = model(chords, future=future)
    return chords_pred
