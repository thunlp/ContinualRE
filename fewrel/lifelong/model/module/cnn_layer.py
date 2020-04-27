import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..base_model import base_model

class cnn_layer(base_model):

    def __init__(self, max_length = 128, input_size = 50, hidden_size=230, type = "cnn"):
        super(cnn_layer, self).__init__()
        self.type = type
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = input_size
        # For CNN and PCNN
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)
        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100
        # For mask length
        seq_ids = torch.arange(self.max_length)
        zeros = torch.zeros([self.max_length, self.max_length])
        ones = torch.ones([self.max_length, self.max_length])
        self.causal_mask = torch.where(seq_ids[None, :].repeat(self.max_length, 1) <= seq_ids[:, None], ones, zeros)
        self.causal_mask = nn.Parameter(self.causal_mask)
        self.causal_mask.requires_grad = False

    def forward(self, inputs = None, length = None, mask = None):
        if self.type.lower() == "cnn":
            return self.cnn(inputs, length)
        else:
            return self.pcnn(inputs, length, mask)

    def cnn(self, inputs, length = None):
        x = self.conv(inputs.transpose(1, 2)) # n x hidden x length
        x = F.relu(x)
        if length == None:
            x = self.pool(x)
        else:
            mask = (1 - torch.index_select(self.causal_mask, 0, length - 1))
            mask = mask.resize(mask.shape[0], 1, mask.shape[1]).float()
            x = self.pool(x + self._minus * mask)
        return x.squeeze(2) # n x hidden_size

    def pcnn(self, inputs, length = None, mask = None):
        x = self.conv(inputs.transpose(1, 2)) # n x hidden x length
        mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
        if length == None:
            pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
            pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
            pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
        else:
            mask_all = (1 - torch.index_select(self.causal_mask, 0, length - 1))
            mask_all = mask.resize(mask.shape[0], 1, mask.shape[1]).float()
            pool1 = self.pool(F.relu(x + self._minus * (mask[:, 0:1, :] + mask_all)))
            pool2 = self.pool(F.relu(x + self._minus * (mask[:, 1:2, :] + mask_all)))
            pool3 = self.pool(F.relu(x + self._minus * (mask[:, 2:3, :] + mask_all)))
        x = torch.cat([pool1, pool2, pool3], 1)
        x = x.squeeze(2) # n x (hidden_size * 3) 