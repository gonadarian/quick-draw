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

        m = tf.shape(mapping)[0]
        vertices = mapping.shape[1]
        regions = mapping.shape[2]
        assert mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

        row_mapping = tf.reshape(tf.range(m), (-1, 1))  # (?, 1)
        row_mapping = tf.tile(row_mapping, [1, vertices * regions])  # (?, 36)
        row_mapping = tf.reshape(row_mapping, (-1, vertices, regions))  # (?, 4, 9)
        assert row_mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

        nodes_mapping = tf.stack([row_mapping, mapping], axis=-1)  # (?, 4, 9, 2)
        assert nodes_mapping.shape[1:] == (vertices, regions, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        non_empty_mask = tf.not_equal(nodes_mapping, empty)  # (?, 4, 9, 2)
        assert non_empty_mask.shape[1:] == (vertices, regions, 2)
        non_empty_mask = tf.reduce_all(non_empty_mask, axis=3)  # (?, 4, 9)
        assert non_empty_mask.shape[1:] == (vertices, regions)
        non_empty_indices = tf.where(non_empty_mask)  # (?, 3)
        assert non_empty_indices.shape[1:] == (3,)

        mapped_nodes_indices = tf.gather_nd(nodes_mapping, non_empty_indices)  # (?, 2)
        assert mapped_nodes_indices.shape[1:] == (2,)

        channels = nodes.shape[2]
        assert nodes.shape[1:] == (vertices, channels)  # (?, 4, 14)

        mapped_nodes = tf.gather_nd(nodes, mapped_nodes_indices)  # (?, 14)
        assert mapped_nodes.shape[1:] == (channels,)

        nodes_columns = tf.scatter_nd(
            indices=non_empty_indices,
            updates=mapped_nodes,
            shape=(m, vertices, regions, channels))
        assert nodes_columns.shape[1:] == (vertices, regions, channels)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, vertices, regions * channels))
        assert nodes_columns.shape[1:] == (vertices, regions * channels)  # (?, 4, 126)

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
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }

        base_config = super(GraphConv, self).get_config()

        return {**base_config, **config}


class Graph2Col(Layer):

    def __init__(self, **kwargs):
        super(Graph2Col, self).__init__(**kwargs)

        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = False
        self.built = False

    def build(self, input_shape):
        mapping_shape = input_shape

        self.input_spec = InputSpec(ndim=3, axes={-1: mapping_shape[-1]})
        self.built = True

    def call(self, inputs, **kwargs):
        mapping = inputs

        m = tf.shape(mapping)[0]
        vertices = mapping.shape[1]
        regions = mapping.shape[2]
        assert mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

        row_mapping = tf.reshape(tf.range(m), (-1, 1))  # (?, 1)
        row_mapping = tf.tile(row_mapping, [1, vertices * regions])  # (?, 36)
        row_mapping = tf.reshape(row_mapping, (-1, vertices, regions))  # (?, 4, 9)
        assert row_mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

        nodes_mapping = tf.stack([row_mapping, mapping], axis=-1)  # (?, 4, 9, 2)
        assert nodes_mapping.shape[1:] == (vertices, regions, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        non_empty_mask = tf.not_equal(nodes_mapping, empty)  # (?, 4, 9, 2)
        assert non_empty_mask.shape[1:] == (vertices, regions, 2)
        non_empty_mask = tf.reduce_all(non_empty_mask, axis=3)  # (?, 4, 9)
        assert non_empty_mask.shape[1:] == (vertices, regions)
        column_indices = tf.where(non_empty_mask)  # (?, 3)
        assert column_indices.shape[1:] == (3,)

        nodes_indices = tf.gather_nd(nodes_mapping, column_indices)  # (?, 2)
        assert nodes_indices.shape[1:] == (2,)

        return [nodes_indices, column_indices]

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3

        # there is no batch sized dim here as the 1st dim depends on graph's edge count
        nodes_indices_shape = (None, 2)
        column_indices_shape = (None, 3)

        return [nodes_indices_shape, column_indices_shape]

    def get_config(self):
        config = {}

        base_config = super(Graph2Col, self).get_config()

        return {**base_config, **config}


class GraphConvV2(Layer):

    def __init__(self, units,
                 regions_size=9,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(GraphConvV2, self).__init__(**kwargs)

        self.units = units
        self.regions = regions_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.input_spec = [
            InputSpec(ndim=3),
            InputSpec(ndim=2, axes={-1: 2}),
            InputSpec(ndim=2, axes={-1: 3}),
        ]

        self.kernel = None
        self.bias = None
        self.built = False

    def build(self, input_shape):
        [nodes_shape, _, _] = input_shape

        channels = nodes_shape[2]

        dense_input_dim = channels * self.regions

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
            InputSpec(ndim=3, axes={-1: channels}),
            InputSpec(ndim=2, axes={-1: 2}),
            InputSpec(ndim=2, axes={-1: 3}),
        ]

        self.built = True

    def call(self, inputs, **kwargs):
        [nodes, nodes_indices, column_indices] = inputs

        m = tf.shape(nodes)[0]
        vertices = nodes.shape[1]
        channels = nodes.shape[2]
        assert nodes.shape[1:] == (vertices, channels)  # (?, 4, 14)

        mapped_nodes = tf.gather_nd(nodes, nodes_indices)  # (?, 14)
        assert mapped_nodes.shape[1:] == (channels,)

        nodes_columns = tf.scatter_nd(
            indices=column_indices,
            updates=mapped_nodes,
            shape=(m, vertices, self.regions, channels))
        assert nodes_columns.shape[1:] == (vertices, self.regions, channels)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, vertices, self.regions * channels))
        assert nodes_columns.shape[1:] == (vertices, self.regions * channels)  # (?, 4, 126)

        output = k.dot(nodes_columns, self.kernel)
        output = k.bias_add(output, self.bias, data_format='channels_last')
        output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        [nodes_shape, _, _] = input_shape
        assert len(nodes_shape) == 3

        output_shape = list(nodes_shape)
        output_shape[-1] = self.units

        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'regions_size': self.regions,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }

        base_config = super(GraphConvV2, self).get_config()

        return {**base_config, **config}
