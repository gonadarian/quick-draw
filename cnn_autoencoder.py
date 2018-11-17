import numpy as np
import models as mdls
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


preload = True
train = False
predict = True
analyze_1 = False
analyze_2 = False


if preload:
    from keras.models import load_model
    autoencoder = load_model('models\lines\model_autoencoder_v2.385-0.0047.hdf5')

else:
    autoencoder = mdls.get_model_autoencoder()
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
    decoded_images = autoencoder.predict(x_test)
    indexes = random.sample(range(1, decoded_images.shape[0]), n)

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
        image_reconstruct = decoded_images[img_idx].reshape(28, 28)
        plt.imshow(image_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(img_original - image_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# show activations of encoded layer with 14 numbers
if analyze_1:
    autoencoder.outputs = [autoencoder.layers[8].output]
    for idx in range(sample_count):
        sample = [x_test[[idx], ...]]
        activation = autoencoder.predict(sample)
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), activation[0,0,0,:]))))


# generate images for encoded values of choice
if analyze_2:
    import matplotlib.pyplot as plt

    # remove layers from encoder part of autoencoder
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
    new_model = Model(input=input, output=x)
    new_model.summary()

    # this is a 14-number encoding for one of the lines in the test set
    sample = np.array([[[[-0.266, 0.209, 0.830, -0.031, 0.069, 0.922, -0.987, 0.800, -0.882, 0.431, 0.853, 0.117, 0.793, 0.388]]]])

    # show 10 images for different encoding variations
    for idx in range(10):
        img = new_model.predict(sample)
        img = img.reshape(28, 28)

        plt.gray()
        plt.imshow(img)
        plt.show()

        # test what happens when 3rd number is increased
        sample[0, 0, 0, 0] += 0.1
