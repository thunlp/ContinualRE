import torch
from torch import nn, optim
from ..base_model import base_model

class softmax_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, id2rel, drop = 0, config = None):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(softmax_layer, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.output_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.id2rel = id2rel
        self.rel2id = {}
        self.drop = nn.Dropout(drop)
        self.config = config
        for id, rel in id2rel.items():
            self.rel2id[rel] = id

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def get_feature(self, sentences, length = None):
        rep = self.sentence_encoder(sentences, length)
        return rep.detach()

    def forward(self, sentences, length = None):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences, length) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        return logits