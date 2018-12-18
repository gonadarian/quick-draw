import time as t
import random as rand
import matplotlib.pyplot as plt
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27

train = True
preload = not train
predict = True
analyze = False


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


def analysis(encoder_model, x):
    m = x.shape[0]

    for idx in range(m):
        sample = [x[[idx], ...]]
        activation = encoder_model.predict(sample)
        print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), activation[0, 0, 0, :]))))


def main(concept, batch_size=64, period=10):

    autoencoder_model, encoder_model, decoder_model = concept.get_model_autoencoder(trained=preload)
    autoencoder_model.summary()

    x, _ = concept.dataset_centered()

    if train:
        epochs = 1000
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
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=period)
            ]
        )

    if predict:
        prediction(autoencoder_model, x)

    # show activations of encoded layer with 14 numbers
    if analyze:
        analysis(encoder_model, x)


if __name__ == '__main__':

    main(Concept.LINE)
    main(Concept.ELLIPSE)
    main(Concept.BEZIER)

    print('end')
