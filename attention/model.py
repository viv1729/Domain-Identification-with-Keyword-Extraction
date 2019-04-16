import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb



class atten_classifier(nn.Module):
    def __init__(self, embed_size, nHidden, nClasses):
        super(atten_classifier, self).__init__()

        self.embed_size = embed_size
        self.nHidden = nHidden

        self.lstm = nn.LSTM(embed_size, nHidden, bidirectional = True)

        self.uw =  nn.Parameter(torch.randn(2*nHidden, 1), requires_grad = True)
        self.hidden2context = nn.Linear(2*nHidden, 2*nHidden)

        self.out_linear = nn.Linear(2*nHidden, nClasses)



    def forward(self, in_seq):
        in_seq = in_seq.view(-1, 1, self.embed_size)
        recurrent, (hidden, c) = self.lstm(in_seq)
        recurrent = recurrent.view(-1, 2*self.nHidden)
        ut = torch.tanh(self.hidden2context(recurrent))
        alphas = torch.softmax(torch.mm(ut, self.uw), 0)

        context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

        out = self.out_linear(context)
        out = out.view(1,-1)
        return out, alphas

# For debugging
# atten_model = atten_classifier(4, 3, 7)
# inp = torch.rand(2, 4)
# output = atten_model(inp)