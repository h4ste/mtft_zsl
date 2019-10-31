import tensorflow.keras as keras
import tensorflow.keras.backend as K


class TiedEmbeddingDecoder(keras.layers.Layer):
    """Layer for tying embeddings in an output layer.

    A regular embedding layer has the shape: V x H (V: size of the vocabulary. H: size of the projected space).
    In this layer, we’ll go: H x V with the same weights than the regular embedding.
    In addition, it may have an activation.

    # References
    - Using the Output Embedding to Improve Language Models:
        https://arxiv.org/abs/1608.05859
    """

    def __init__(self,
                 encoder: keras.layers.Embedding,
                 activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.activation = keras.activations.get(activation)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not self.encoder.built:
            assert len(input_shape) == 2
            self.encoder.build(input_shape[::-1])
        self.embeddings_transpose = K.transpose(self.encoder.weights[0])
        self.built = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0], K.int_shape(self.encoder.weights[0])[0]

    def call(self, inputs, mask=None):
        output = K.dot(inputs, self.transposed_weights)
        if self.activation is not None:
            output = self.activation(output)
        return output
