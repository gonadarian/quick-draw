import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.models import Model
from keras.optimizers import Adam


# representation (14, 1, 1) i.e. 14-dimensional
def get_model_large_2():

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


# representation (14, 1, 1) i.e. 14-dimensional
def get_model_large_1():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(4, (3, 3), activation='relu', padding='valid')(input_img)
    x = Conv2D(8, (5, 5), activation='relu', padding='valid')(x)
    x = Conv2D(12, (7, 7), activation='relu', padding='valid')(x)
    x = Conv2D(16, (7, 7), activation='relu', padding='valid')(x)
    x = Conv2D(20, (5, 5), activation='relu', padding='valid')(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(28, (4, 4), activation='relu', padding='valid')(x)
    x = Conv2D(14, (1, 1), activation='relu', padding='valid')(x)
    encoded = x
    x = Conv2D(20, (1, 1), activation='relu', padding='same')(x)  # !!!!!!!!!!!! not used!!! :(
    x = UpSampling2D((3, 3))(encoded)
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
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, decoded)
    return model


# representation (4, 4, 1) i.e. 16-dimensional
def get_model_conv3_16_8_1():
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, decoded)
    return model


# representation (4, 4, 8) i.e. 128-dimensional
def get_model_conv3_16_8_8():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, decoded)
    return model


preload = True
train = False
predict = True
analyze_1 = False
analyze_2 = False


if preload:
    from keras.models import load_model
    # autoencoder = load_model('models\large-2.297-0.0112.hdf5')
    autoencoder = load_model('models\lines\model_autoencoder_v2.385-0.0047.hdf5')

else:
    autoencoder = get_model_large_2()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


print(autoencoder.summary())


x_train = np.load('generator\lines_392x28x28_v2.npy')
x_test = np.load('generator\lines_392x28x28_v2.npy')
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
sample_count = x_test.shape[0]


if train:
    from keras.callbacks import TensorBoard, ModelCheckpoint

    autoencoder.fit(
        x_train, x_train,
        epochs=400,
        batch_size=64,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[
            TensorBoard(log_dir='C:\Logs'),
            ModelCheckpoint(
                'models\lines\model_autoencoder_v2.{epoch:02d}-{val_loss:.4f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ]
    )


if predict:
    import matplotlib.pyplot as plt
    import random

    n = 10
    decoded_imgs = autoencoder.predict(x_test)
    indexes = random.sample(range(1, decoded_imgs.shape[0]), n)

    plt.figure(figsize=(30, 4))
    for i in range(n):
        img_idx = indexes[i]
        # display original
        ax = plt.subplot(3, n, i + 1)
        img_original = x_test[img_idx].reshape(28, 28)
        plt.imshow(img_original)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        img_reconstruct = decoded_imgs[img_idx].reshape(28, 28)
        plt.imshow(img_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(img_original - img_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# show activations of encoded layer with 14 numbers
if analyze_1:
    autoencoder.outputs = [autoencoder.layers[8].output]
    for idx in range(sample_count):
        sample = [x_test[[idx],...]]
        activation = autoencoder.predict(sample)
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), activation[0,0,0,:]))))


# generate images for encoded values of choice
if analyze_2:
    import matplotlib.pyplot as plt

    # remove layers from encoder part of autoencoder
    for i in range(9):
        autoencoder.layers.pop(0)

    # add new input layer to represent encoded state with 14 numbers
    input = Input(shape=(1,1,14))

    # relink all the layers again to include new input one in the chain
    x = input
    layers = [l for l in autoencoder.layers]
    for i in range(len(layers)):
        x = layers[i](x)

    # create new model with this new layer chain
    new_model = Model(input=input, output=x)
    new_model.summary()

    # this is a 14-number encoding for one of the lines in the test set
    # sample = np.array([[[[0.00, 0.00, 0.00, 231.35, 351.92, 0.00, 578.25, 0.00, 0.00, 153.61, 177.18, 58.08, 410.07, 0.00]]]])
    sample = np.array([[[[-0.266, 0.209, 0.830, -0.031, 0.069, 0.922, -0.987, 0.800, -0.882, 0.431, 0.853, 0.117, 0.793, 0.388]]]])

    # show 10 images for different encoding variations
    for idx in range(10):
        img = new_model.predict(sample)
        img = img.reshape(28, 28)

        plt.gray()
        plt.imshow(img)
        plt.show()

        # test what happens when 3rd number is increased
        sample[0,0,0,0] += 0.1

