import torch
from torch.autograd import Variable
import torch.nn.init as init

from tensorboardX import SummaryWriter

from utils.params import parse_arguments, default_params

import numpy as np


def main(**params):
	params = dict(
        default_params,
        **params
    )

    # Set random seed
    np.random.seed(params['seed'])

    # Create writer for tensorboard: Visualize plots at real time when training
    # TODO: Put each log_dir inside the experiment
	writer = SummaryWriter(log_dir='tboard')

	dtype = torch.FloatTensor
	input_size, hidden_size, output_size = 7, 6, 1
	epochs = 300
	seq_length = 20
	lr = 0.1

	data_time_steps = np.linspace(2, 10, seq_length + 1)
	data = np.sin(data_time_steps)
	data.resize((seq_length + 1, 1))

	x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
	y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)

	w1 = torch.FloatTensor(input_size, hidden_size).type(dtype)
	init.normal_(w1, 0.0, 0.4)
	w1 =  Variable(w1, requires_grad=True)
	w2 = torch.FloatTensor(hidden_size, output_size).type(dtype)
	init.normal_(w2, 0.0, 0.3)
	w2 = Variable(w2, requires_grad=True)

	def forward(input, context_state, w1, w2):
		xh = torch.cat((input, context_state), 1)
		context_state = torch.tanh(xh.mm(w1))
		out = context_state.mm(w2)
		return  (out, context_state)

	for i in range(epochs):
		context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
		total_loss = 0
		for j in range(x.size(0)):
			input = x[j:(j+1)]
			target = y[j:(j+1)]
			(pred, context_state) = forward(input, context_state, w1, w2)
			loss = (pred - target).pow(2).sum()/2
			total_loss += loss
			loss.backward()
			w1.data -= lr * w1.grad.data
			w2.data -= lr * w2.grad.data
			w1.grad.data.zero_()
			w2.grad.data.zero_()
			context_state = Variable(context_state.data)
		writer.add_scalar('data/loss', total_loss, i)
		if i % 10 == 0:
		 print("Epoch: {} loss {}".format(i, total_loss.item()))

		context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
		predictions = []

		for i in range(x.size(0)):
			input = x[i:i+1]
			(pred, context_state) = forward(input, context_state, w1, w2)
			context_state = context_state
			predictions.append(pred.data.numpy().ravel()[0])

	writer.close()


if __name__ == '__main__':
    main(**vars(parse_arguments()))
