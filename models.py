from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D


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
    input = Input(shape=(1, 1, 14))

    # relink all the layers again to include new input one in the chain
    x = input
    layers = [l for l in autoencoder.layers]
    for i in range(len(layers)):
        x = layers[i](x)

    # create new model with this new layer chain
    decoder = Model(inputs=input, outputs=x)

    if show:
        decoder.summary()

    return decoder
