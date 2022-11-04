import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.initializers import RandomUniform

def AE():
    x_input = Input(shape=[28, 28, 1])

    # x = Conv2D(8, 3, 1, 'same')(x_input)

    x = Flatten()(x_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu', name='latent')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(784, activation='relu')(x)
    x = Reshape((28, 28, 1))(x)

    model = Model(x_input, x)
    model.summary()

    return model

def MemAE():
    x_input = Input(shape=[28, 28, 1])

    # x = Conv2D(8, 3, 1, 'same')(x_input)

    x = Flatten()(x_input) # 784
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = MemoryUnit(100)(x) # 64
    x = Dense(256, activation='relu')(x)
    x = Dense(784, activation='relu')(x)
    x = Reshape((28, 28, 1))(x)

    model_2 = Model(x_input, x)
    return model_2



class MemoryUnit(Layer):
    def __init__(self, mem_dim, shrink_thres=0.0025):
        # C: dimension of vector z
        # M: size of the memory
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.kernel_regularizer = None
        self.shrink_thres = shrink_thres

    def build(self, input_shape):
        self.std = 8

        # M x C
        self.weight = self.add_weight(shape=(self.mem_dim, input_shape[-1]),
                                      initializer=RandomUniform(-self.std, self.std, seed=2803),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

    def call(self, inputs):
        # att_weight = F.linear(inputs, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        # att_weight = F.softmax(att_weight, dim=1)  # TxM

        att_weight = compute_cosine_distances(inputs, self.weight)  # Fea x Mem^T, (batchxTxC) x (CxM) = TxM
        # att_weight = tf.matmul(inputs, tf.transpose(self.weight))
        att_weight = softmax(att_weight)  # TxM

        if (self.shrink_thres > 0):
            att_weight = relu(att_weight, threshold=self.shrink_thres)

            # normalize by p=1 (L1 normalization)
            att_weight, _ = tf.linalg.normalize(att_weight, ord=1, axis=1)
        output = tf.matmul(att_weight, self.weight)
        return output

def compute_cosine_distances(a, b):
    # a: Input, shape = (batch * n_a * fea_dim)
    # b: Memory, shape = (n_b * fea_dim)

    # output: shape = (batch * n_a * n_b)

    a_normalized, _ = tf.linalg.normalize(a, ord=1, axis=-1)
    b_normalized, _ = tf.linalg.normalize(b, ord=1, axis=-1)

    # b_normalized_transposed = tf_swap_last_2_axis(b_normalized)
    b_normalized_transposed = tf.transpose(b_normalized)

    distance = tf.matmul(a_normalized, b_normalized_transposed)

    return distance