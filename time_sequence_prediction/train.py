from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

import json

VOCAB_SIZE = 283
EMBED_SIZE = 7
EPOCHS = 500

class Sequence(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, embed_size, input_size=1):
        super(Sequence, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embed_size = embed_size

        self.embed = torch.nn.Embedding(self.vocabulary_size, self.embed_size) # 2 words in vocab, 5 dimensional embeddings
        self.lstm1 = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.vocabulary_size) # make it with our vocabulary size


    def forward(self, input, future = 0):
        outputs = []
        batch_size = input.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)


        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            x_embedded = self.embed(input_t.view(-1).long())

            h_t, c_t = self.lstm1(x_embedded, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)

            outputs += [output]

        for i in range(future): # if we should predict the future
            output = torch.argmax(output, dim=1).view(-1, 1).double()

            output = self.embed(output.view(-1).long())

            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def show_embedding(self, writer):

        with open('../data/mappings.json', "r") as infile:
            mapping = json.load(infile)

        input = torch.FloatTensor(list(mapping['chord'].values()))
        labels = list(mapping['chord'].keys())


        writer.add_embedding(self.embed(input.long()), metadata=labels)


if __name__ == '__main__':

    print('*' * 20 + ' Start Harmonic Sequence Predictor ' + '*' * 20)

    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    data_train = np.load('../data/train_sequential_clean_data.npy')
    data_train = data_train[:, :, 0] # take only the chords
    input = torch.from_numpy(data_train[:, :-1])
    target = torch.from_numpy(data_train[:, 1:])
    data_val = np.load('../data/validation_sequential_clean_data.npy')
    data_val = data_val[:, :, 0] # take only the chords
    val_input = torch.from_numpy(data_val[:, :-1])
    val_target = torch.from_numpy(data_val[:, 1:])
    data_test = np.load('../data/test_sequential_clean_data.npy')
    data_test = data_test[:, :, 0]  # take only the chords
    test_input = torch.from_numpy(data_test[:, :-1])
    test_target = torch.from_numpy(data_test[:, 1:])

    # tensorboard
    writer = SummaryWriter(log_dir='../results/tboard/embeddings')

    # build the model
    seq = Sequence(283, 51, 7)
    seq.double()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    optimizer = optim.Adam(seq.parameters(), lr=0.01)

    #begin to train
    for i in range(EPOCHS):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out.view(-1, out.size(2)), target.contiguous().view(-1).long())
            print('loss:', loss.item())
            loss.backward()

            # tensorboard
            writer.add_scalar('train_loss', loss, i)

            return loss
        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 10
            pred = seq(val_input)
            y_pred = pred.contiguous().view(-1, pred.size(2))
            y = val_target.contiguous().view(-1).long()
            loss = criterion(y_pred, y)

            # tensorboard
            writer.add_scalar('val_loss', loss, i)

            print('val loss:', loss.item())
            y = pred.detach().numpy()

    seq.show_embedding(writer)
    writer.close()

    with torch.no_grad():
        pred = seq(test_input)
        y_pred = pred.contiguous().view(-1, pred.size(2))
        y = test_target.contiguous().view(-1).long()
        loss = criterion(y_pred, y)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
