import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self,out,keep_probab,maxstrlen, embedding_length, vocab_size):
        super(MLP, self).__init__()
        self.out = out
        self.maxstrlen = maxstrlen
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.embedding = nn.Embedding(vocab_size,embedding_length)
        self.fc1 = nn.Linear(300,100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100,10)
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(maxstrlen*10,128)
        self.fc4 = nn.Linear(128, out)
        self.dropout = nn.Dropout(keep_probab)


    def forward(self,s):
        x = self.embedding(s)
        f1 = self.fc1(x)
        f2 = self.fc2(self.relu1(f1))
        f2 = f2.view(-1, self.maxstrlen * 10)
        f3 = self.fc3(self.relu2(f2))
        f3 = self.dropout(f3)
        logit = self.fc4(self.relu3(f3))
        logit = F.softmax(logit, dim=1)


        return logit

