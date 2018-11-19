import torch


def forward(input_, context_state, w1, w2):
    xh = torch.cat((input_, context_state), 1)
    context_state = torch.tanh(xh.mm(w1))
    out = context_state.mm(w2)
    return out, context_state
