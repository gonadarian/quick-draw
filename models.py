import os
from keras.layers import Input, Conv2D, UpSampling2D, Dense
from keras.models import Model, load_model


def load(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    model = load_model(path)
    return model


def load_autoencoder_model():
    autoencoder_model = load('models\lines\lines_autoencoder_v2-385-0.0047.hdf5')
    return autoencoder_model


def load_encoder_model():
    encoder_model = load('models\lines_encoded\lines_mixed_encoded_v2-091-0.000072.hdf5')
    return encoder_model


def load_decoder_model():
    autoencoder_model = load_autoencoder_model();
    decoder_model = gen_decoder_model(autoencoder_model)
    return decoder_model


def load_clustering_model():
    clustering_model = load('models\pairs_encoded\model_dense_v1-088-0.001555.hdf5')
    return clustering_model


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
    layers = [l for l in autoencoder.layers]
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
