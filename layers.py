import tensorflow as tf
from keras import backend as k
from keras import activations, initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec


class GraphConv(Layer):

    def __init__(self, units,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphConv, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.supports_masking = True

        self.kernel = None
        self.bias = None
        self.built = False

    def build(self, input_shape):
        [nodes_shape, mapping_shape] = input_shape
        dense_input_shape = (nodes_shape[0], nodes_shape[1], nodes_shape[2] * mapping_shape[2])
        dense_input_dim = dense_input_shape[-1]

        self.kernel = self.add_weight(shape=(dense_input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.input_spec = [
            InputSpec(ndim=3, axes={-1: nodes_shape[-1]}),
            InputSpec(ndim=3, axes={-1: mapping_shape[-1]})
        ]

        self.built = True

    def call(self, inputs, **kwargs):
        [nodes, mapping] = inputs

        batch_size = tf.shape(nodes)[0]
        node_size = nodes.shape[1]
        channel_size = nodes.shape[2]
        region_size = mapping.shape[2]
        assert nodes.shape[1:] == (node_size, channel_size)  # (?, 4, 14)
        assert mapping.shape[1:] == (node_size, region_size)  # (?, 4, 9)

        row_mapping = tf.reshape(tf.range(batch_size), (-1, 1))  # (?, 1)
        row_mapping = tf.tile(row_mapping, [1, node_size * region_size])  # (?, 36)
        row_mapping = tf.reshape(row_mapping, (-1, node_size, region_size))  # (?, 4, 9)
        assert row_mapping.shape[1:] == (node_size, region_size)  # (?, 4, 9)

        nodes_mapping = tf.stack([row_mapping, mapping], axis=-1)  # (?, 4, 9, 2)
        assert nodes_mapping.shape[1:] == (node_size, region_size, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        non_empty_mask = tf.not_equal(nodes_mapping, empty)  # (?, 4, 9, 2)
        assert non_empty_mask.shape[1:] == (node_size, region_size, 2)
        non_empty_mask = tf.reduce_all(non_empty_mask, axis=3)  # (?, 4, 9)
        assert non_empty_mask.shape[1:] == (node_size, region_size)
        non_empty_indices = tf.where(non_empty_mask)  # (?, 3)
        assert non_empty_indices.shape[1:] == (3,)

        mapped_nodes_indices = tf.gather_nd(nodes_mapping, non_empty_indices)  # (?, 2)
        assert mapped_nodes_indices.shape[1:] == (2,)

        mapped_nodes = tf.gather_nd(nodes, mapped_nodes_indices)  # (?, 14)
        assert mapped_nodes.shape[1:] == (channel_size,)

        nodes_columns = tf.scatter_nd(
            indices=non_empty_indices,
            updates=mapped_nodes,
            shape=(batch_size, node_size, region_size, channel_size))
        assert nodes_columns.shape[1:] == (node_size, region_size, channel_size)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, node_size, region_size * channel_size))
        assert nodes_columns.shape[1:] == (node_size, region_size * channel_size)  # (?, 4, 126)

        output = k.dot(nodes_columns, self.kernel)
        output = k.bias_add(output, self.bias, data_format='channels_last')
        output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units

        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConv, self).get_config()

        return {**base_config, **config}


