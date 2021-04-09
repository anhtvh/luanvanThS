from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.lstm = nn.LSTM(embedding_length, hidden_size,bidirectional=True,dropout=0.5,num_layers=2)
        self.label = nn.Linear(hidden_size*4, output_size)


    def forward(self, input_sentence, batch_size=None):
        input = input_sentence.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        output, (final_hidden_state, final_cell_state) = self.lstm(input)

        final = torch.cat((final_hidden_state[0],final_hidden_state[1],final_hidden_state[2],final_hidden_state[3]),dim=1)
        final_output = self.label(final)  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        final_output = F.softmax(final_output,dim=1)

        return final_output
