import abc

import tensorflow as tf
import tensorflow.keras as keras

import fslks as fsl


class QAModel(keras.Model):

    def __init__(self, transformer, vocab_size, embedding_size, *args, **kwargs):
        # todo: implement/wrap a transformer
        super().__init__(*args, **kwargs)
        self.vocab_encoder = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
        self.vocab_decoder = fsl.layers.TiedEmbeddingDecoder(self.vocab_encoder)

