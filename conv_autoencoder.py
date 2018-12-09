import time as t
import numpy as np
import random as rand
import libs.models as mdls
import matplotlib.pyplot as plt
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27

train = True
preload = not train
predict = True
analyze_1 = False
analyze_2 = False


def prediction(autoencoder_model, x, n=10):
    m = x.shape[0]

    predicted_list = autoencoder_model.predict(x)
    indices = rand.sample(range(1, m), n)
    plt.figure(figsize=(30, 4))

    for i in range(n):
        index = indices[i]

        # display original
        ax = plt.subplot(3, n, i + 1)
        original = x[index].reshape(dim, dim)
        plt.imshow(original)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(index)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        predicted = predicted_list[index].reshape(dim, dim)
        plt.imshow(predicted)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(original - predicted)
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
    decoder_model = mdls.extract_decoder_model(autoencoder_model, show=True)

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


def main(concept):

    autoencoder_model = concept.model_autoencoder() if preload else concept.model_autoencoder_creator()
    autoencoder_model.summary()

    x, _ = concept.dataset_centered()

    if train:
        epochs = 1000
        batch_size = 64
        timestamp = int(t.time())

        model_name = 'conv-autoencoder-{}-{}'.format(concept.code, timestamp)
        log_dir = 'C:\Logs\{}-b{}'.format(model_name, batch_size)
        filepath = 'models\{}\{}-{}.hdf5'.format(concept.code, model_name, 'e{epoch:04d}-{val_loss:.5f}')

        autoencoder_model.fit(
            x, x,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x, x),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=10)
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
    main(Concept.LINE)
    main(Concept.ELLIPSE)
    print('end')
