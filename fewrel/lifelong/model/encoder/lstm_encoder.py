from .base_encoder import base_encoder
from ..module import embedding_layer, lstm_layer

class lstm_encoder(base_encoder):

    def __init__(self, token2id = None, word2vec = None, word_size = 50, max_length = 128, 
            pos_size = None, hidden_size = 230, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        super(lstm_encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding = False)
        self.config = config
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_size = word_size
        self.pos_size = pos_size
        self.input_size = word_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        if pos_size != None:
            self.input_size += 2 * pos_size
        self.embedding_layer = embedding_layer(self.word2vec, max_length, word_size, None, False)
        self.encoder_layer = lstm_layer(max_length, self.input_size, hidden_size, dropout, bidirectional, num_layers, config)

    def forward(self, inputs, lengths = None):
        inputs, lengths, inputs_indexs = self.encoder_layer.pad_sequence(inputs, padding_value = self.token2id['[PAD]'])
        inputs = inputs.to(self.config['device'])
        x = self.embedding_layer(inputs)
        x = self.encoder_layer(x, lengths, inputs_indexs)
        return x