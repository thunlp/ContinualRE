import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..base_model import base_model

class lstm_layer(base_model):

    def __init__(self, max_length = 128, input_size = 50, hidden_size = 256, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        """
        Args:
            input_size: dimention of input embedding
            hidden_size: hidden size
            dropout: dropout layer on the outputs of each RNN layer except the last layer
            bidirectional: if it is a bidirectional RNN
            num_layers: number of recurrent layers
            activation_function: the activation function of RNN, tanh/relu
        """
        super(lstm_layer, self).__init__()
        self.device = config['device']
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers, dropout = dropout)

    def init_hidden(self, batch_size = 1, device='cpu'):
        self.hidden = (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))
    
    def forward(self, inputs, lengths, inputs_indexs):
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        lstm_out, hidden = self.lstm(packed_embeds, self.hidden)
        permuted_hidden = hidden[0].permute([1,0,2]).contiguous()
        permuted_hidden = permuted_hidden.view(-1, self.hidden_size * 2)
        output_embedding = permuted_hidden[inputs_indexs]
        return output_embedding

    def ranking_sequence(self, sequence):
        word_lengths = torch.tensor([len(sentence) for sentence in sequence])
        rankedi_word, indexs = word_lengths.sort(descending = True)
        ranked_indexs, inverse_indexs = indexs.sort()
        sequence = [sequence[i] for i in indexs]
        return sequence, inverse_indexs
    
    def pad_sequence(self, inputs, padding_value = 0):
        self.init_hidden(len(inputs), self.device)
        inputs, inputs_indexs = self.ranking_sequence(inputs)
        lengths = [len(data) for data in inputs]
        pad_inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value = padding_value)
        return pad_inputs, lengths, inputs_indexs
