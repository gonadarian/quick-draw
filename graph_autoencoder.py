import keras.backend as K
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Conv2D, UpSampling2D, Dense, Average, RepeatVector

node_count = 4
edge_count = 4
region_count = 9
encoding_dim = 17


def graph2col_shape(input_shapes):
    [nodes_shape, mapping_shape] = input_shapes
    batch_size = nodes_shape[0]
    node_size = nodes_shape[1]
    node_encoding_size = nodes_shape[2]
    region_size = mapping_shape[0] // node_size
    column_size = node_encoding_size * region_size

    return batch_size, node_size, column_size


def graph2col(inputs):
    [nodes, mapping] = inputs
    assert nodes.shape[1:] == (node_count, encoding_dim)
    assert mapping.shape[1:] == (node_count * region_count, )
    nodes = K.reshape(nodes, (..., encoding_dim))
    mapping = K.reshape(mapping, (..., ))
    nodes_columns = K.gather(nodes, mapping)
    assert nodes_columns.shape[1:] == (node_count * region_count, encoding_dim)
    nodes_columns = nodes_columns.reshape((node_count, node_count * region_count))
    assert nodes_columns.shape[1:] == (node_count, region_count * encoding_dim)

    return nodes_columns


def get_model_autoencoder():

    input_nodes = Input(shape=(node_count, encoding_dim))
    input_mapping = Input(shape=(node_count * region_count, ), dtype='int32')

    x = input_nodes

    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(20, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(26, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(32, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(26, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(20, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(14, activation='tanh')(x)
    # x = Average

    # encoded = x

    # x = RepeatVector
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(20, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(26, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(32, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(26, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(20, activation='relu')(x)
    x = Lambda(graph2col, output_shape=graph2col_shape)([x, input_mapping])
    x = Dense(14, activation='sigmoid')(x)

    decoded = x

    model = Model([[input_nodes, input_mapping]], decoded)
    return model


def get_nodes_columsn(nodes, vector_indexes):
    nodes_columns = K.gather(nodes, vector_indexes)
    assert nodes_columns.shape == (node_count * (region_count + 1), encoding_dim)
    assert nodes_columns.shape == (4 * 9, 17)
    nodes_columns = nodes_columns.reshape((node_count, (region_count + 1) * encoding_dim))
    assert nodes_columns.shape == (4, 9 * 17)

    return nodes_columns


def main():
    # nodes = np.zeros((node_count, encoding_dim))
    # assert nodes.shape == (4, 17)
    # vector_indexes = np.zeros((node_count * (region_count + 1), ))
    # assert vector_indexes.shape == (4 * 9, )
    #
    # nodes_columns = get_nodes_columsn(nodes, vector_indexes)
    # print('to columns:', nodes_columns)

    model = get_model_autoencoder()
    model.summary()


if __name__ == '__main__':
    main()
    print('end')
