import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..base_model import base_model

class embedding_layer(base_model):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim = 50, pos_embedding_dim = None, requires_grad = True):
        super(embedding_layer, self).__init__()
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        
        # Word embedding
        # unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        # blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx = word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)
        self.word_embedding.weight.requires_grad = requires_grad
        # Position Embedding
        if self.pos_embedding_dim != None:
            self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx = 0)
            self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx = 0)

    def forward(self, word, pos1 = None, pos2 = None):
        if pos1 != None and pos2 != None and self.pos_embedding_dim != None:
            x = torch.cat([self.word_embedding(word), 
                            self.pos1_embedding(pos1), 
                            self.pos2_embedding(pos2)], 2)
        else:
            x = self.word_embedding(word)
        return x


