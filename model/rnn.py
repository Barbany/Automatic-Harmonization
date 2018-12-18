import sys
import torch
import torch.nn as nn
import json


class Sequence(torch.nn.Module):
    def __init__(self, vocabulary_size, hidden_size, embedding, embed_size, cond_size=None, input_rnn_size=1):
        super(Sequence, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.embed_size = embed_size
        self.input_rnn_size = input_rnn_size
        self.cond_size = cond_size

        if self.cond_size is None:
            if self.embedding:
                # Embedding without conditioner
                self.input_rnn_size = embed_size
            else:
                # Floating point values of the chord mappings
                self.input_rnn_size = 1

        if self.embedding:
            self.embed = torch.nn.Embedding(self.vocabulary_size, self.embed_size) # 2 words in vocab, 5 dimensional embeddings
            if cond_size is not None:
                self.embed_expand = torch.nn.Linear(
                    in_features=self.embed_size,
                    out_features=self.input_rnn_size
                )

                self.cond_expand = torch.nn.Linear(
                    in_features=self.cond_size,
                    out_features=self.input_rnn_size
                )

        self.lstm1 = nn.LSTMCell(self.input_rnn_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.vocabulary_size)  # make it with our vocabulary size


    def forward(self, input, features, future = 0):
        outputs = []
        batch_size = input.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)

        for idx, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            if self.embedding:
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

            if self.embedding:
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
