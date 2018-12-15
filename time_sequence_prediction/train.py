from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import json

VOCAB_SIZE = 282
EMBED_SIZE = 7
EPOCHS = 500

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, EMBED_SIZE) # 2 words in vocab, 5 dimensional embeddings
        self.lstm1 = nn.LSTMCell(EMBED_SIZE, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, VOCAB_SIZE) # make it with our vocabulary size

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            x_embedded = self.embed(input_t.view(-1).long())

            h_t, c_t = self.lstm1(x_embedded, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)

            outputs += [output]
        for i in range(future):# if we should predict the future
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
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    # tensorboard
    writer = SummaryWriter(log_dir='../results/tboard/embeddings')

    # build the model
    seq = Sequence()
    seq.double()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
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
            pred = seq(test_input, future=future)
            y_pred = pred[:, :-future].contiguous().view(-1, pred.size(2))
            y = test_target.contiguous().view(-1).long()
            loss = criterion(y_pred, y)
            writer.add_scalar('test_loss', loss, i)
            print('test loss:', loss.item())
            y = pred.detach().numpy()

    seq.show_embedding(writer)
    writer.close()