from .base_encoder import base_encoder
from ..module import embedding_layer, cnn_layer

class cnn_encoder(base_encoder):

    def __init__(self, token2id, word2vec, word_size = 50, max_length = 128, 
            pos_size = None, hidden_size = 230):
        super(cnn_encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding = True)

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_size = word_size
        self.pos_size = pos_size
        self.input_size = word_size
        if pos_size != None:
            self.input_size += 2 * pos_size
        self.embedding_layer = embedding_layer(self.word2vec, max_length, word_size, pos_embedding_dim, False)
        self.encoder_layer = cnn_layer(max_length, self.input_size, hidden_size, "cnn")

    def forward(self, inputs, length = None):
        x = self.embedding(inputs)
        x = self.encoder(x, length)
        return x