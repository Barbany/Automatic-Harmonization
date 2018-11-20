import sys
import torch

class RNN(torch.nn.Module):

    def __init__(self, vocabulary_size, embedding_size, hidden_size, use_cuda, layers=1, dropout_prob=0,
                 bidirectional=False, batch_first=False):
        """
        Recurrent Neural Network with LSTM
        :param vocabulary_size: Number of different chords
        :param embedding_size: Dimensionality of embedding of chords - Also input size of RNN (Hypothesis: 7)
        :param hidden_size: Number of features in the hidden state
        :param layers : Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
                        together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM
                        and computing the final results. Default: 1
        :param dropout_prob: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                             with dropout probability equal to dropout. Default: 0
        :param bidirectional: If True, becomes a bidirectional LSTM. Default: False
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        """
        super(RNN, self).__init__()

        # Configuration of our model
        self.num_layers = layers
        self.hidden_size = hidden_size
        self.cuda = use_cuda

        # Define embedding layer
        self.embed = torch.nn.Embedding(vocabulary_size, embedding_size)

        # Define LSTM
        self.lstm = torch.nn.LSTM(embedding_size, self.hidden_size, self.num_layers, dropout=dropout_prob,
                                batch_first=batch_first, bidirectional=bidirectional)

        # Define dropout
        self.drop = torch.nn.Dropout(dropout_prob)

        # Define output layer: Fully Connected
        self.fc = torch.nn.Linear(self.hidden_size,vocabulary_size)

        # Init weights
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range,init_range)

        return

    def forward(self, x, h):
        # Apply embedding (encoding)
        y = self.embed(x)
        # Run LSTM
        y = self.drop(y)
        y, h = self.lstm(y,h)
        y = self.drop(y)
        # Reshape
        y = y.contiguous().view(-1,self.hidden_size)
        # Fully-connected (decoding)
        y = self.fc(y)
        # Return prediction and states
        return y,h

    def get_initial_states(self, batch_size):
        # Set initial hidden and memory states to 0
        if self.cuda:
            return (torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)).cuda(),
                    torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)),
                    torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)))

    def detach(self, h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(),h[1].detach()
