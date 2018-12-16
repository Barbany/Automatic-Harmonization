import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

import json

class Sequence(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, embed_size=None, cond_size=None, input_rnn_size=1):
        super(Sequence, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.input_rnn_size = input_rnn_size
        self.cond_size = cond_size

        if self.cond_size is None:
            if self.embed_size is None:
                # Floating point values of the chord mappings
                self.input_rnn_size = 1
            else:
                # Embedding without conditioner
                self.input_rnn_size = embed_size

        if embed_size is not None:
            self.embed = torch.nn.Embedding(self.vocabulary_size, self.embed_size) # 2 words in vocab, 5 dimensional embeddings
            if cond_size is not None:
                self.embed_expand = torch.nn.Linear(
                    in_features=self.embed_size,
                    out_features=self.input_rnn_size
                )

                self.cond_expand = torch.nn.Linear(
                    in_features=cond_size,
                    out_features=input_rnn_size
                )
        self.lstm1 = nn.LSTMCell(self.input_rnn_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.vocabulary_size) # make it with our vocabulary size


    def forward(self, input, features, future = 0):
        outputs = []
        batch_size = input.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)


        for idx, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            if self.embed_size is not None:
                input_t = self.embed(input_t.view(-1).long())
                if self.cond_size is not None:
                    embed_expanded = self.embed_expand(input_t)
                    cond_expanded = self.cond_expand(features[:, idx])
                    input_t = embed_expanded + cond_expanded

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)

            outputs += [output]

        for i in range(future): # if we should predict the future
            output = torch.argmax(output, dim=1).view(-1, 1).double()
            
            if self.embed_size is not None:
                output = self.embed(output.view(-1).long())
                if self.cond_size is not None:
                    embed_expanded = self.embed_expand(output)
                    cond_expanded = self.cond_expand(features[:, idx + i + 1])
                    output = embed_expanded + cond_expanded

            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def show_embedding(self, mapping, writer):

        input = torch.FloatTensor(list(mapping['chord'].values()))
        labels = list(mapping['chord'].keys())


        writer.add_embedding(self.embed(input.long()), metadata=labels)


if __name__ == '__main__':

    print('*' * 20 + ' Start Harmonic Sequence Predictor ' + '*' * 20)

    params = {
        'hidden_size': 51,
        'embedding': 7,
        'epochs': 500,
        'conditioner': True,
        'input_rnn_size': 40,
        'Tm': 50, # Maximum number of iterations.
        'lr_controller': True,
        'gradient_clip': 0.25
    }

    np.random.seed(123)
    torch.manual_seed(123)

    # load data and make training set
    data_train = np.load('../data/train_sequential_clean_data.npy')
    input = torch.from_numpy(data_train[:, :-1])
    target = torch.from_numpy(data_train[:, 1:, 0])
    data_val = np.load('../data/validation_sequential_clean_data.npy')
    val_input = torch.from_numpy(data_val[:, :-1])
    val_target = torch.from_numpy(data_val[:, 1:, 0])
    data_test = np.load('../data/test_sequential_clean_data.npy')
    test_input = torch.from_numpy(data_test[:, :-1])
    test_target = torch.from_numpy(data_test[:, 1:, 0])

    mapping_file = '../data/mappings.json'
    with open(mapping_file, "r") as infile:
        mapping = json.load(infile)

    vocabulary_size = len(mapping['chord']) + 1
    
    if params['conditioner']:
        num_features = data_train.shape[2] - 1
    else:
        num_features = None

    # tensorboard
    writer = SummaryWriter(log_dir='../results/tboard/embeddings')

    # build the model
    seq = Sequence(vocabulary_size=vocabulary_size, 
                   hidden_size=params['hidden_size'],
                   embed_size=params['embedding'],
                   cond_size=num_features,
                   input_rnn_size=params['input_rnn_size']
                )

    seq.double()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    optimizer = optim.Adam(seq.parameters(), lr=0.01)

    if params['lr_controller']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params['Tm'], eta_min=0)

    #begin to train
    for i in range(params['epochs']):
        print('STEP: ', i)

        def closure():
            if params['lr_controller']:
                scheduler.step()
            optimizer.zero_grad()
            out = seq(input[:, :, 0], input[:, :, 1:])
            loss = criterion(out.view(-1, out.size(2)), target.contiguous().view(-1).long())
            print('loss:', loss.item())
            loss.backward()

            if params['gradient_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(seq.parameters(), params['gradient_clip'])

            # tensorboard
            writer.add_scalar('train_loss', loss, i)

            return loss
        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 10
            pred = seq(val_input[:, :, 0], val_input[:, :, 1:])
            y_pred = pred.contiguous().view(-1, pred.size(2))
            y = val_target.contiguous().view(-1).long()
            loss = criterion(y_pred, y)

            # tensorboard
            writer.add_scalar('val_loss', loss, i)

            print('val loss:', loss.item())
            y = pred.detach().numpy()

    seq.show_embedding(mapping, writer)
    writer.close()

    with torch.no_grad():
        pred = seq(test_input[:, :, 0], test_input[:, :, 1:])
        y_pred = pred.contiguous().view(-1, pred.size(2))
        y = test_target.contiguous().view(-1).long()
        loss = criterion(y_pred, y)
        print('test loss:', loss.item())
        y = pred.detach().numpy()
