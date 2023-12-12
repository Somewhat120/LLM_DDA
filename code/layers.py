import tensorflow as tf
from utils import *

def weight_variable_glorot(input_dim, output_dim, name=''):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010) initialization."""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors."""
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)
    
class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse.sparse_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class GraphConvolution_2(tf.keras.layers.Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu):
        super(GraphConvolution_2, self).__init__()
        self.vars = {}
        self.issparse = False
        self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = tf.cast(adj, tf.float32)
        self.act = act

    def call(self, inputs, training=False):
        x = inputs
        if training:
            x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs
    
class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse.sparse_dense_matmul(x, self.vars['weights'])
            x = tf.sparse.sparse_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs

class GraphConvolutionSparse_2(tf.keras.layers.Layer):
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu):
        super(GraphConvolutionSparse_2, self).__init__()
        self.vars = {}
        self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = tf.cast(adj, tf.float32)
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
    
    def call(self, inputs, training=False):
        x = inputs
        if training:
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse.sparse_dense_matmul(x, self.vars['weights'])
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class AttentionAggregator(tf.keras.layers.Layer):
    def __init__(self, num_vectors):
        super(AttentionAggregator, self).__init__()
        initial_value = tf.random.uniform([num_vectors], 0, 1)
        self.attention_weights = tf.Variable(
            initial_value=initial_value,
            trainable=True,
            name="attention_weights"
        )

    def call(self, inputs):
        # 计算注意力权重的 softmax
        attention_scores = tf.nn.softmax(self.attention_weights)
        # 聚合向量
        aggregated_vector = tf.reduce_sum(inputs * tf.reshape(attention_scores, [-1, 1, 1]), axis=0)

        return aggregated_vector
    

class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs

class InnerProductDecoder_2(tf.keras.layers.Layer):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, num_r, dropout=0., act=tf.nn.sigmoid):
        super(InnerProductDecoder_2, self).__init__()
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def call(self, inputs, training=False):
        if training:
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
        R = inputs[0:self.num_r, :]
        D = inputs[self.num_r:, :]
        R = tf.matmul(R, self.vars['weights'])
        D = tf.transpose(D)
        x = tf.matmul(R, D)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
