import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
import numpy as np
from ..base_model import base_model
from ...utils.tokenization import WordTokenizer

class base_encoder(base_model):

    def __init__(self, 
                 token2id = None, 
                 word2vec = None,
                 word_size = 50,
                 max_length = 128,
                 blank_padding = True):
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
        super(base_encoder, self).__init__()

        if isinstance(token2id, list):
            self.token2id = {}
            for index, token in enumerate(token2id):
                self.token2id[token] = index
        else:
            self.token2id = token2id

        self.max_length = max_length
        self.num_token = len(self.token2id)

        if isinstance(word2vec, type(None)):
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]

        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        if not isinstance(word2vec, type(None)):
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:            
                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word2vec = (torch.cat([word2vec, unk, blk], 0)).numpy()
            else:
                self.word2vec = word2vec
        else:
            self.word2vec = None

        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")
    
    def set_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer
    
    def set_encoder_layer(self, encoder_layer):
        self.encoder_layer = encoder_layer 

    def forward(self, token, pos1, pos2):
        pass
    
    def tokenize(self, sentence):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            index number of tokens and positions             
        """
        tokens = self.tokenizer.tokenize(sentence)
        length = min(len(tokens), self.max_length)
        # Token -> index

        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.token2id['[UNK]'])

        if (len(indexed_tokens) > self.max_length):
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        length = torch.tensor([length]).long()
        return indexed_tokens, length
