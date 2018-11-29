import os
import lambdas as ls
import tensorflow as tf
import keras.backend as k
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, UpSampling2D, Dense, Lambda
from layers import GraphConv, GraphConvV2, Graph2Col


def load(filename, custom_objects=None):
    path = os.path.join(os.path.dirname(__file__), 'models/', filename)
    model = load_model(path, custom_objects)
    return model


def load_autoencoder_model():
    autoencoder_model = load('lines/lines_autoencoder_v2-385-0.0047.hdf5')
    return autoencoder_model


def load_encoder_model():
    encoder_model = load('lines_encoded/lines_mixed_encoded_v2-091-0.000072.hdf5')
    return encoder_model


def load_decoder_model():
    autoencoder_model = load_autoencoder_model()
    decoder_model = gen_decoder_model(autoencoder_model)
    return decoder_model


def load_clustering_model():
    clustering_model = load('pairs_encoded/model_dense_v1-088-0.001555.hdf5')
    return clustering_model


def load_graph_autoencoder_model(node_count, region_count, version=1):
    custom_objects = {
        'tf': tf,
        'node_count': node_count,
        'region_count': region_count,
        'GraphConv': GraphConv,
    }
    versions = {
        1: 'graphs/model_autoencoder_v1.09500-0.000011.hdf5',
        2: 'graphs/model_autoencoder_v2.08800-0.000009.hdf5',
        3: 'graphs/model_autoencoder_v3.09700-0.000008.hdf5',
    }

    autoencoder_model = load(versions[version], custom_objects)
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'acc'])

    return autoencoder_model


# representation (14, 1, 1) i.e. 14-dimensional
def get_model_autoencoder():

    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(4, (3, 3), activation='relu', padding='valid')(input_img)
    x = Conv2D(8, (5, 5), activation='relu', padding='valid')(x)
    x = Conv2D(12, (7, 7), activation='relu', padding='valid')(x)
    x = Conv2D(16, (7, 7), activation='relu', padding='valid')(x)
    x = Conv2D(20, (5, 5), activation='relu', padding='valid')(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(28, (4, 4), activation='relu', padding='valid')(x)
    x = Conv2D(14, (1, 1), activation='tanh', padding='valid')(x)

    encoded = x

    x = Conv2D(20, (1, 1), activation='relu', padding='same')(encoded)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(28, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(20, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(12, (7, 7), activation='relu', padding='same')(x)
    x = Conv2D(8, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoded = x

    model = Model(input_img, decoded)
    return model


def gen_decoder_model(autoencoder, show=False):
    # remove layers from encoder part of auto-encoder
    for i in range(9):
        autoencoder.layers.pop(0)

    # add new input layer to represent encoded state with 14 numbers
    x = Input(shape=(1, 1, 14))

    # relink all the layers again to include new input one in the chain
    y = x
    layers = [layer for layer in autoencoder.layers]
    for i in range(len(layers)):
        y = layers[i](y)

    # create new model with this new layer chain
    decoder = Model(inputs=x, outputs=y)

    if show:
        decoder.summary()

    return decoder


def create_encoder_model():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(12, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(20, (7, 7), activation='relu', padding='same')(x)
    x = Conv2D(24, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(28, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(22, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(17, (3, 3), activation='tanh', padding='same')(x)

    encoded = x

    model = Model(input_img, encoded)
    return model


def create_clustering_model():
    input_data = Input(shape=(34,))

    x = input_data
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    output_data = x
    model = Model(inputs=input_data, outputs=output_data)

    return model


def absolute_loss(vector):
    loss = k.mean(vector)
    assert loss.shape == ()
    return loss


def create_graph_autoencoder_model(node_count, encoding_dim, region_count, version=1):
    versions = {
        1: _create_graph_autoencoder_model_v1,
        2: _create_graph_autoencoder_model_v2,
        3: _create_graph_autoencoder_model_v3,
    }
    create_model = versions[version]
    model = create_model(node_count, encoding_dim, region_count)

    return model


def _create_graph_autoencoder_model_v1(node_count, encoding_dim, region_count):
    input_nodes = Input(shape=(node_count, encoding_dim), name='Input_Nodes')
    input_graph = Input(shape=(node_count, region_count), dtype='int32', name='Input_Graph')

    encoding = input_nodes

    encoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Enc_1')([encoding, input_graph])
    encoding = Dense(20, activation='relu', name='Dense_Enc_1')(encoding)
    encoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Enc_2')([encoding, input_graph])
    encoding = Dense(26, activation='relu', name='Dense_Enc_2')(encoding)
    encoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Enc_3')([encoding, input_graph])
    encoding = Dense(32, activation='relu', name='Dense_Enc_3')(encoding)
    encoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Enc_4')([encoding, input_graph])
    encoding = Dense(26, activation='relu', name='Dense_Enc_4')(encoding)
    encoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Enc_5')([encoding, input_graph])
    encoding = Dense(20, activation='relu', name='Dense_Enc_5')(encoding)
    encoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Enc_6')([encoding, input_graph])
    encoding = Dense(14, activation='tanh', name='Dense_Enc_6')(encoding)

    encoding, variance = Lambda(ls.lambda_moments, ls.lambda_moments_shape, name='Encoding_Moments')(encoding)
    decoding = encoding
    decoding = Lambda(ls.lambda_repeat, ls.lambda_repeat_shape, name='Encoding_Repeats')(decoding)

    decoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Dec_1')([decoding, input_graph])
    decoding = Dense(20, activation='relu', name='Dense_Dec_1')(decoding)
    decoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Dec_2')([decoding, input_graph])
    decoding = Dense(26, activation='relu', name='Dense_Dec_2')(decoding)
    decoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Dec_3')([decoding, input_graph])
    decoding = Dense(32, activation='relu', name='Dense_Dec_3')(decoding)
    decoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Dec_4')([decoding, input_graph])
    decoding = Dense(26, activation='relu', name='Dense_Dec_4')(decoding)
    decoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Dec_5')([decoding, input_graph])
    decoding = Dense(20, activation='relu', name='Dense_Dec_5')(decoding)
    decoding = Lambda(ls.lambda_graph2col, ls.lambda_graph2col_shape, name='Graph_Dec_6')([decoding, input_graph])
    decoding = Dense(14, activation='tanh', name='Dense_Dec_6')(decoding)

    model = Model(inputs=[input_nodes, input_graph], outputs=decoding)
    model.add_loss(absolute_loss(variance))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def _create_graph_autoencoder_model_v2(node_count, encoding_dim, region_count):
    input_nodes = Input(shape=(node_count, encoding_dim), name='Input_Nodes')
    input_graph = Input(shape=(node_count, region_count), dtype='int32', name='Input_Graph')

    encoding = input_nodes

    encoding = GraphConv(20, name='GraphConv_Enc_1')([encoding, input_graph])
    encoding = GraphConv(26, name='GraphConv_Enc_2')([encoding, input_graph])
    encoding = GraphConv(32, name='GraphConv_Enc_3')([encoding, input_graph])
    encoding = GraphConv(26, name='GraphConv_Enc_4')([encoding, input_graph])
    encoding = GraphConv(20, name='GraphConv_Enc_5')([encoding, input_graph])
    encoding = GraphConv(14, activation='tanh', name='GraphConv_Enc_6')([encoding, input_graph])

    encoding, variance = Lambda(ls.lambda_moments, ls.lambda_moments_shape, name='Encoding_Moments')(encoding)
    decoding = encoding
    decoding = Lambda(ls.lambda_repeat, ls.lambda_repeat_shape, name='Encoding_Repeats')(decoding)

    decoding = GraphConv(20, name='GraphConv_Dec_1')([decoding, input_graph])
    decoding = GraphConv(26, name='GraphConv_Dec_2')([decoding, input_graph])
    decoding = GraphConv(32, name='GraphConv_Dec_3')([decoding, input_graph])
    decoding = GraphConv(26, name='GraphConv_Dec_4')([decoding, input_graph])
    decoding = GraphConv(20, name='GraphConv_Dec_5')([decoding, input_graph])
    decoding = GraphConv(14, activation='tanh', name='GraphConv_Dec_6')([decoding, input_graph])

    model = Model(inputs=[input_nodes, input_graph], outputs=decoding)
    model.add_loss(absolute_loss(variance))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def _create_graph_autoencoder_model_v3(node_count, encoding_dim, region_count):
    input_graph = Input(shape=(node_count, region_count), dtype='int32', name='Input_Graph')
    input_nodes = Input(shape=(node_count, encoding_dim), name='Input_Nodes')

    with k.name_scope('Prepare'):
        nodes_indices, column_indices = Graph2Col(name='Graph2Col')(input_graph)

    encoding = input_nodes

    with tf.name_scope('Encoder'):
        encoding = GraphConvV2(20, name='GraphConv_Enc_1')([encoding, nodes_indices, column_indices])
        encoding = GraphConvV2(26, name='GraphConv_Enc_2')([encoding, nodes_indices, column_indices])
        encoding = GraphConvV2(32, name='GraphConv_Enc_3')([encoding, nodes_indices, column_indices])
        encoding = GraphConvV2(26, name='GraphConv_Enc_4')([encoding, nodes_indices, column_indices])
        encoding = GraphConvV2(20, name='GraphConv_Enc_5')([encoding, nodes_indices, column_indices])
        encoding = GraphConvV2(14, activation='tanh', name='GraphConv_Enc_6')([encoding, nodes_indices, column_indices])

    with tf.name_scope('Embedding'):
        encoding, variance = Lambda(ls.lambda_moments, ls.lambda_moments_shape, name='Encoding_Moments')(encoding)
        decoding = encoding
        decoding = Lambda(ls.lambda_repeat, ls.lambda_repeat_shape, name='Encoding_Repeats')(decoding)

    with tf.name_scope('Decoder'):
        decoding = GraphConvV2(20, name='GraphConv_Dec_1')([decoding, nodes_indices, column_indices])
        decoding = GraphConvV2(26, name='GraphConv_Dec_2')([decoding, nodes_indices, column_indices])
        decoding = GraphConvV2(32, name='GraphConv_Dec_3')([decoding, nodes_indices, column_indices])
        decoding = GraphConvV2(26, name='GraphConv_Dec_4')([decoding, nodes_indices, column_indices])
        decoding = GraphConvV2(20, name='GraphConv_Dec_5')([decoding, nodes_indices, column_indices])
        decoding = GraphConvV2(14, activation='tanh', name='GraphConv_Dec_6')([decoding, nodes_indices, column_indices])

    model = Model(inputs=[input_nodes, input_graph], outputs=decoding)
    model.add_loss(absolute_loss(variance))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

    return model
