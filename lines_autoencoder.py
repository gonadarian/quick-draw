import time as t
import numpy as np
import random as rand
import libs.models as mdls
import libs.datasets as ds
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27

preload = True
train = not preload
predict = True
analyze_1 = False
analyze_2 = False


def prediction(autoencoder_model, x):
    n = 10
    m = x.shape[0]

    decoded_images = autoencoder_model.predict(x)
    indexes = rand.sample(range(1, m), n)
    plt.figure(figsize=(30, 4))

    for i in range(n):
        img_idx = indexes[i]

        # display original
        ax = plt.subplot(3, n, i + 1)
        img_original = x[img_idx].reshape(dim, dim)
        plt.imshow(img_original)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        image_reconstruct = decoded_images[img_idx].reshape(dim, dim)
        plt.imshow(image_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(img_original - image_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def analysis_1(autoencoder_model, x):
    m = x.shape[0]

    autoencoder_model.outputs = [autoencoder_model.layers[8].output]

    for idx in range(m):
        sample = [x[[idx], ...]]
        activation = autoencoder_model.predict(sample)
        print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), activation[0, 0, 0, :]))))


def analysis_2(autoencoder_model):
    decoder_model = mdls.gen_decoder_model(autoencoder_model, show=True)

    # this is a 14-number encoding for one of the lines in the test set
    sample = np.array([[[
        [-0.266, 0.209, 0.830, -0.031, 0.069, 0.922, -0.987, 0.800, -0.882, 0.431, 0.853, 0.117, 0.793, 0.388]
    ]]])

    # show 10 images for different encoding variations
    for idx in range(10):
        img = decoder_model.predict(sample)
        img = img.reshape(dim, dim)

        plt.gray()
        plt.imshow(img)
        plt.show()

        # test what happens when 3rd number is increased
        sample[0, 0, 0, 0] += 0.1


def main():

    if preload:
        autoencoder_model = mdls.load_autoencoder_model_27x27()

    else:
        autoencoder_model = mdls.get_model_autoencoder_27x27()
        autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder_model.summary()

    x = ds.load_images_line_27x27_centered()

    if train:
        autoencoder_model.fit(
            x, x,
            epochs=1000,
            batch_size=64,
            shuffle=True,
            validation_data=(x, x),
            callbacks=[
                TensorBoard(log_dir='C:\Logs\Conv Autoencoder v4.b64.{}'.format(int(t.time()))),
                ModelCheckpoint(
                    'models\lines\model_autoencoder_v3.{epoch:04d}-{val_loss:.5f}.hdf5',
                    monitor='val_loss', verbose=0, save_best_only=True,
                    save_weights_only=False, mode='auto', period=10)
            ]
        )

    if predict:
        prediction(autoencoder_model, x)

    # show activations of encoded layer with 14 numbers
    if analyze_1:
        analysis_1(autoencoder_model, x)

    # generate images for encoded values of choice
    if analyze_2:
        analysis_2(autoencoder_model)


if __name__ == '__main__':
    main()
    print('end')
