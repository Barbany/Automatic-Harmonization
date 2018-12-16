import sys
import torch

class Sequence(torch.nn.Module):
    def __init__(self, hidden_size, vocabulary_size, input_size=1):
        # Modify input size if we are dealing with embeddings
        super(Sequence, self).__init__()

        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size

        self.lstm1 = torch.nn.LSTMCell(input_size=input_size, 
                                      hidden_size=self.hidden_size)
        self.lstm2 = torch.nn.LSTMCell(input_size=hidden_size, 
                                      hidden_size=self.hidden_size)
        self.linear = torch.nn.Linear(self.hidden_size, self.vocabulary_size)

        # Define softmax layer to convert output of FC into probabilities (Convert to pdf along dimension 0)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input, writer=None, future = 0):
        outputs = []
        batch_size = input.size(0)

        h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)

        for input_t in input.chunk(input.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            output = self.softmax(output)
            outputs += [output]
        for _ in range(future):# if we should predict the future
            output = torch.argmax(output, dim=1).view(-1, 1).double()
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            output = self.softmax(output)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class RNN(torch.nn.Module):

    def __init__(self, vocabulary_size, embedding_size, num_features, rnn_input_size, hidden_size, writer,
                 use_cuda, layers=3, dropout_prob=0, bidirectional=False, batch_first=False):
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
        self.vocabulary_size = vocabulary_size
        
        # Visualization with Tensorboard
        self.writer = writer

        # Define embedding layer
        self.embed = torch.nn.Embedding(vocabulary_size, embedding_size)

        self.embed_expand = torch.nn.Conv1d(
            in_channels=embedding_size,
            out_channels=rnn_input_size,
            kernel_size=1
        )

        self.cond_expand = torch.nn.Conv1d(
            in_channels=num_features,
            out_channels=rnn_input_size,
            kernel_size=1
        )

        # Define LSTM
        self.lstm = torch.nn.LSTM(
            rnn_input_size,
            self.hidden_size,
            self.num_layers,
            dropout=dropout_prob,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

        # Define output layer: Fully Connected
        # self.writer.add_embedding(features, metadata=x[0])
        self.fc = torch.nn.Linear(self.hidden_size,vocabulary_size)

        # Define softmax layer to convert output of FC into probabilities
        self.softmax = torch.nn.Softmax(1)

        # Init weights
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range,init_range)

        return

    def forward(self, x, cond, h):
        # Internal level of verbosity. Only suggested for debugging purposes
        verbose = False

        # Apply embedding (encoding). Data type has to be casted to long before it
        # Size: [batch_size, chord_seq_len, embedding_size]
        x_embedded = self.embed(x.long())
        if verbose:
            print('\n', '*'*60)
            print('Chords before embedding have size', x.size())
            print('Chords after embedding have size', x_embedded.size())

        # Apply 1D-Convolution to both the embedding and the features
        # Input size is of the form [batch_size, channels_in, length_in]
        # ** Note that embedding has length and channels reversed - Need to permute **
        # Output size: [batch_size, channels_out, length_out]
        x_input_rnn = self.embed_expand(x_embedded.permute(0, 2, 1)).permute(0, 2, 1)
        cond_input_rnn = self.cond_expand(cond.float().permute(0, 2, 1)).permute(0, 2, 1)
        if verbose:
            print('Features before 1D-Conv have size', cond.size())
            print('Features after 1D-Conv have size', cond_input_rnn.size())
            print('Chords after 1D-Conv have size', x_input_rnn.size())

        # Add features to condition input of RNN
        x_input_rnn += cond_input_rnn
        
        # Run LSTM
        y, h = self.lstm(x_input_rnn, h)
        if verbose:
            print('Output of LSTM has size', y.size())

        # Reshape
        y = y.contiguous().view(-1, self.hidden_size)

        # Fully-connected (decoding)
        y = self.fc(y)
        if verbose:
            print('Output of Fully Connected layer has size', y.size())

        # Apply Softmax layer
        y = self.softmax(y.view(1, -1, self.vocabulary_size))

        # Return prediction (most probable class in form of float) and states.
        y_pred = y.view(-1, self.vocabulary_size)
        if verbose:
            print(y_pred)
        return y_pred, h

    ## Functions with internal calls of pytorch library
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
