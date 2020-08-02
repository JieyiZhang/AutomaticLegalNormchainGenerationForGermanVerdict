from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers.core import Layer

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class LabelwiseAttention(Layer):

    def __init__(self, kernel_regularizer=None, bias_regularizer=None,
                 return_attention=False, n_classes=4271, **kwargs):

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)
        self.init = initializers.get('he_normal')
        self.supports_masking = True
        self.return_attention = return_attention
        self.n_classes = n_classes
        super(LabelwiseAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.Wa = self.add_weight(shape=(self.n_classes, input_shape[-1]),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  name='{}_Wa'.format(self.name))

        self.Wo = self.add_weight(shape=(self.n_classes, input_shape[-1]),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  name='{}_Wo'.format(self.name))

        self.bo = self.add_weight(shape=(self.n_classes,),
                                  initializer='zeros',
                                  regularizer=self.b_regularizer,
                                  name='{}_bo'.format(self.name))

        super(LabelwiseAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):

        a = dot_product(x, self.Wa)

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = K.softmax(K.transpose(ai))
            label_aware_doc_rep = K.dot(ai, doc_repi)
            if self.return_attention:
                return [label_aware_doc_rep, ai]
            else:
                return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores =  K.tensorflow_backend.map_fn(label_wise_attention, [x, a])

        # Compute label-scores
        label_aware_doc_reprs = K.sum(label_aware_doc_reprs * self.Wo, axis=-1) + self.bo
        label_aware_doc_reprs = K.sigmoid(label_aware_doc_reprs)

        if self.return_attention:
            return [label_aware_doc_reprs, attention_scores]

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], self.n_classes),
                    (input_shape[0], input_shape[1], self.n_classes, input_shape[-1])]
        return input_shape[0], self.n_classes

