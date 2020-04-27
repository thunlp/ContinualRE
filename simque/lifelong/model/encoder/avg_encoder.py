import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .base_encoder import base_encoder

class avg_encoder(base_encoder):

    def __init__(self, 
                 token2id = None, 
                 word2vec = None,
                 word_size = 50,
                 max_length = 128,
                 blank_padding = True,
                 dropout=0):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
        """
        # hyperparameters
        super(avg_encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding)
        self.drop = nn.Dropout(dropout)
        self.max_length = max_length
        self.num_token = len(token2id)

        seq_ids = torch.arange(self.max_length)
        zeros = torch.zeros([self.max_length, self.max_length])
        ones = torch.ones([self.max_length, self.max_length])
        self.causal_mask = torch.where(seq_ids[None, :].repeat(self.max_length, 1) <= seq_ids[:, None], ones, zeros)
        self.causal_mask = nn.Parameter(self.causal_mask)
        self.causal_mask.requires_grad = False

    def forward(self, token, length):
        """
        Args:
            token: (B, L), index of tokens
            length: (B, L) length of tokens
        Return:
            (B, H), (B, L, H), representations for sentences & hidden states
        """
        # Check size of tensors
        x = self.embedding_layer(token)
        x = self.drop(x)
        mask = torch.index_select(self.causal_mask, 0, length - 1)
        length = length.resize(length.shape[0], 1).float()
        avg = torch.sum(x * mask[:, :, None], 1) / length
        return avg, x

    def predict(self, token, length):
        avg, x = self.forward(token, length)
        return avg.cpu().data.numpy(), x.cpu().data.numpy()

    def tokenize(self, sentence):
        return super().tokenize(sentence)
