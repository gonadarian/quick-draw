import os
import tensorflow as tf
import keras.backend as k
import libs.lambdas as ls
import keras.losses as loss
import keras.metrics as metric
import keras.optimizers as optimizer
from libs.losses import absolute_loss
from libs.layers import GraphConv, Graph2Col
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, UpSampling2D, Dense, Lambda


def load(filename, custom_objects=None):
    path = os.path.join(os.path.dirname(__file__), '../models/', filename)
    model = load_model(path, custom_objects)
    print('LOADED MODEL from path {}'.format(path))
    return model


def load_autoencoder_line_model_27x27():
    autoencoder_model = load('line/conv-autoencoder-line-1544197342-e0950-0.00108.hdf5')
    decoder_model = extract_decoder_model(autoencoder_model)
    encoder_model = extract_encoder_model(autoencoder_model)
    return autoencoder_model, encoder_model, decoder_model


def load_matrix_encoder_line_model_27x27():
    matrix_encoder_model = load('line/conv-matrix-encoder-line-1544485883-e0187-0.000034.hdf5')
    return matrix_encoder_model


def load_clustering_line_model():
    clustering_model = load('line/conv-clustering-line-1544542116-e0295-0.001054.hdf5')
    return clustering_model


# TODO refactor to return encoder model as well
def load_autoencoder_ellipse_model_27x27():
    autoencoder_model = load('ellipse/conv-autoencoder-ellipse-1544114775-e0980-0.00196.hdf5')
    decoder_model = extract_decoder_model(autoencoder_model)
    return autoencoder_model, decoder_model


def load_matrix_encoder_ellipse_model_27x27():
    matrix_encoder_model = load('ellipse/conv-matrix-encoder-ellipse-1544229426-e0301-0.002483.hdf5')
    return matrix_encoder_model


def load_clustering_ellipse_model():
    clustering_model = load('ellipse/conv-clustering-ellipse-1544360371-e0920-0.001776.hdf5')
    return clustering_model


def load_graph_autoencoder_model():
    custom_objects = {
        'tf': tf,
        'Graph2Col': Graph2Col,
        'GraphConv': GraphConv,
    }

    autoencoder_model = load('square/graph-autoencoder-square-1544607431-e09900-0.000003.hdf5', custom_objects)
    autoencoder_model.compile(
        optimizer=optimizer.Adam(),
        loss=loss.binary_crossentropy,
        metrics=[metric.mean_absolute_error, metric.binary_accuracy])

    # TODO extract encoder and decoder models as well

    return autoencoder_model


def extract_decoder_model(autoencoder, show=False):
    # add new input layer to represent encoded state with 14 numbers
    x = Input(shape=(1, 1, 14))

    # relink all the layers again to include new input one in the chain
    y = x
    for layer in autoencoder.layers[9:]:
        y = layer(y)

    # create new model with this new layer chain
    decoder = Model(inputs=x, outputs=y)

    if show:
        decoder.summary()

    return decoder


def extract_encoder_model(autoencoder, show=False):
    x = autoencoder.input
    y = autoencoder.layers[8].output

    # create new model with this new layer chain
    encoder = Model(inputs=x, outputs=y)

    if show:
        encoder.summary()

    return encoder


def create_autoencoder_model_27x27():
    input_image = Input(shape=(27, 27, 1))

    encoding = input_image

    encoding = Conv2D(4, (3, 3), activation='relu', padding='valid', name='enc_conv1')(encoding)   # 25
    encoding = Conv2D(8, (5, 5), activation='relu', padding='valid', name='enc_conv2')(encoding)   # 21
    encoding = Conv2D(12, (7, 7), activation='relu', padding='valid', name='enc_conv3')(encoding)  # 15
    encoding = Conv2D(16, (7, 7), activation='relu', padding='valid', name='enc_conv4')(encoding)  # 9
    encoding = Conv2D(20, (5, 5), activation='relu', padding='valid', name='enc_conv5')(encoding)  # 5
    encoding = Conv2D(24, (3, 3), activation='relu', padding='valid', name='enc_conv6')(encoding)  # 3
    encoding = Conv2D(28, (3, 3), activation='relu', padding='valid', name='enc_conv7')(encoding)  # 1
    encoding = Conv2D(14, (1, 1), activation='tanh', padding='valid', name='enc_conv8')(encoding)  # 1

    decoding = encoding

    decoding = Conv2D(28, (1, 1), activation='relu', padding='same', name='dec_conv1')(decoding)  # 1
    decoding = UpSampling2D((3, 3), name='dec_up1')(decoding)
    decoding = Conv2D(24, (3, 3), activation='relu', padding='same', name='dec_conv2')(decoding)  # 1
    decoding = UpSampling2D((3, 3), name='dec_up2')(decoding)
    decoding = Conv2D(20, (3, 3), activation='relu', padding='same', name='dec_conv3')(decoding)  # 1
    decoding = Conv2D(16, (5, 5), activation='relu', padding='same', name='dec_conv4')(decoding)  # 1
    decoding = UpSampling2D((3, 3), name='dec_up3')(decoding)
    decoding = Conv2D(12, (7, 7), activation='relu', padding='same', name='dec_conv5')(decoding)  # 1
    decoding = Conv2D(8, (7, 7), activation='relu', padding='same', name='dec_conv6')(decoding)  # 1
    decoding = Conv2D(4, (5, 5), activation='relu', padding='same', name='dec_conv7')(decoding)  # 1
    decoding = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='dec_conv8')(decoding)  # 1

    output_image = decoding

    autoencoder_model = Model(input_image, output_image)
    autoencoder_model.compile(
        optimizer=optimizer.Adam(),
        loss=loss.binary_crossentropy)

    decoder_model = extract_decoder_model(autoencoder_model)
    encoder_model = extract_encoder_model(autoencoder_model)

    return autoencoder_model, encoder_model, decoder_model


def create_matrix_encoder_model_27x27():
    input_image = Input(shape=(27, 27, 1))

    x = input_image

    # TODO refactor as a loop with method parameters controlling the layers' params
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='enc_conv1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv2')(x)
    x = Conv2D(12, (5, 5), activation='relu', padding='same', name='enc_conv3')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same', name='enc_conv4')(x)
    x = Conv2D(20, (7, 7), activation='relu', padding='same', name='enc_conv5')(x)
    x = Conv2D(24, (5, 5), activation='relu', padding='same', name='enc_conv6')(x)
    x = Conv2D(28, (5, 5), activation='relu', padding='same', name='enc_conv7')(x)
    x = Conv2D(22, (3, 3), activation='relu', padding='same', name='enc_conv8')(x)
    x = Conv2D(17, (3, 3), activation='tanh', padding='same', name='enc_conv9')(x)

    encoded = x

    model = Model(input_image, encoded)
    model.compile(
        optimizer=optimizer.Adam(),
        loss=loss.mean_squared_error,
        metrics=[metric.mean_squared_error, metric.categorical_accuracy])

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
    model.compile(
        optimizer=optimizer.Adam(),
        loss=loss.binary_crossentropy)

    return model


def create_graph_autoencoder_model(units_list, node_count, region_count):
    assert len(units_list) >= 2

    input_units = units_list[0]
    embedding_units = units_list[-1]

    input_graph = Input(shape=(node_count, region_count), dtype='int32', name='Input_Graph')
    input_nodes = Input(shape=(node_count, input_units), name='Input_Nodes')

    with k.name_scope('Prepare'):
        nodes_indices, column_indices = Graph2Col(name='Graph2Col')(input_graph)

    encoding = input_nodes

    with tf.name_scope('Encoder'):
        for layer, units in enumerate(units_list[1:-2:1]):
            encoding = GraphConv(units, name='GraphConv_Enc_{}'.format(layer + 1))([
                encoding,
                nodes_indices,
                column_indices])

        # TODO merge this into the above loop
        encoding = GraphConv(embedding_units, activation='tanh', name='GraphConv_Enc_{}'.format(layer + 2))([
            encoding,
            nodes_indices,
            column_indices])

    with tf.name_scope('Embedding'):
        encoding, variance = Lambda(ls.lambda_moments, ls.lambda_moments_shape, name='Encoding_Moments')(encoding)
        decoding = encoding
        decoding = Lambda(ls.lambda_repeat, ls.lambda_repeat_shape, name='Encoding_Repeats')(decoding)

    with tf.name_scope('Decoder'):
        for layer, units in enumerate(units_list[-2:1:-1]):
            decoding = GraphConv(units, name='GraphConv_Dec_{}'.format(layer + 1))([
                decoding,
                nodes_indices,
                column_indices])

        # TODO merge this into the above loop
        decoding = GraphConv(input_units, activation='tanh', name='GraphConv_Dec_{}'.format(layer + 2))([
            decoding,
            nodes_indices,
            column_indices])

    model = Model(inputs=[input_nodes, input_graph], outputs=decoding)
    model.add_loss(absolute_loss(variance))
    model.compile(
        optimizer=optimizer.Adam(),
        loss=loss.mean_squared_error,
        metrics=[metric.mean_squared_error, metric.categorical_accuracy])

    return model
