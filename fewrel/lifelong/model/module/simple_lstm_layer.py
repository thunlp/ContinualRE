import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..base_model import base_model

class simple_lstm_layer(base_model):

    def __init__(self, max_length = 128, input_size = 256, output_size = 256, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        """
        Args:
            input_size: dimention of input embedding
            hidden_size: hidden size
            dropout: dropout layer on the outputs of each RNN layer except the last layer
            bidirectional: if it is a bidirectional RNN
            num_layers: number of recurrent layers
            activation_function: the activation function of RNN, tanh/relu
        """
        super(simple_lstm_layer, self).__init__()
        self.device = config['device']
        self.max_length = max_length
        self.output_size = output_size
        self.input_size = input_size
        self.config = config
        if bidirectional:
            self.hidden_size = output_size // 2
        else:
            self.hidden_size = output_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional = bidirectional, num_layers = num_layers, dropout = dropout)

    def init_hidden(self, batch_size = 1, device='cpu'):
        self.hidden = (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))
    
    def forward(self, inputs, lengths):
        self.init_hidden(inputs.shape[0], self.config['device'])
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first = True)
        lstm_out, hidden = self.lstm(packed_embeds, self.hidden)
        permuted_hidden = hidden[0].permute([1,0,2]).contiguous()
        output_embedding = permuted_hidden.view(-1, self.hidden_size * 2)
        return output_embedding
