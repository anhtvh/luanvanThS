# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, batch_size,vocab_size , output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab,
                  embedding_length):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.embedding_length = embedding_length
        self.embedding  = nn.Embedding(vocab_size, embedding_length)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length),padding=(kernel_heights[0] - 1, 0))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length),padding=(kernel_heights[1] - 1, 0))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length),padding=(kernel_heights[2] - 1, 0))

        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)
        self.sofmax = nn.Softmax(dim=1)

    def conv_block(self, input, conv_layer):

        conv_out = conv_layer(input)
        activation = torch.squeeze(F.relu(conv_out),-1)
        max_out = F.max_pool1d(activation, activation.size(2))

        return max_out

    def forward(self, x, batch_size=None):
        input_sentences = self.embedding(x)
        input = torch.unsqueeze(input_sentences, 1)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 2)
        all_out = all_out.view(all_out.size(0), -1) 
        fc_in = self.dropout(all_out)
        logits = self.label(fc_in)
        logits = self.sofmax(logits)


        return logits