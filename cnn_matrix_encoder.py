import time as t
import numpy as np
import models as mdls
import random as rand
import utilities as utl
import matplotlib.pyplot as plt
from scipy import spatial
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27
channels = 14
channels_full = 17

preload = True
train = not preload
predict = True
analysis = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_distances(encodings):
    count = encodings.shape[0]
    distances = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            distances[i, j] = spatial.distance.cosine(encodings[i, :], encodings[j, :])

    for idx in range(count):
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), distances[idx, :]))), '\n')

    return distances


def load_data():
    # TODO use datasets load method
    filename = 'generator\data\lines_27x27\line_27x27_mixed_samples_v1_5815x{}x{}x1.npy'
    x = np.load(filename.format(dim, dim))
    assert x.shape[1:] == (dim, dim, 1)
    m = x.shape[0]
    x = np.reshape(x, (m, dim, dim, 1))
    # TODO use datasets load method
    filename = 'generator\data\lines_27x27\line_27x27_mixed_encodings_v1_5815x{}x{}x{}.npy'
    y = np.load(filename.format(dim, dim, channels_full))
    assert y.shape[1:] == (dim, dim, 17)
    assert y.shape[0] == m
    return x, y, m


def main():

    x, y, m = load_data()

    if preload:
        encoder_model = mdls.load_encoder_model_27x27()

    else:
        encoder_model = mdls.create_encoder_model_27x27()
        encoder_model.compile(optimizer='adam', loss='mean_squared_error')

    if train:
        encoder_model.fit(
            x, y,
            epochs=1000,
            batch_size=64,
            shuffle=True,
            validation_data=(x, y),
            callbacks=[
                TensorBoard(log_dir='C:\Logs\Conv Embedder v1.b64.{}'.format(int(t.time()))),
                ModelCheckpoint(
                    'models\lines_27x27\model_encoder_v1-{epoch:03d}-{val_loss:.6f}.hdf5',
                    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            ])

    if predict:
        predict_single = False
        predict_multiple = True

        decoder_model = mdls.load_decoder_model_27x27()

        if predict_single:

            idx = rand.randint(0, x.shape[0])
            print('random index:', idx)
            sample = x[idx, :, :, 0]
            embeddings = utl.get_embeddings(encoder_model, sample, dim=dim, show=True)

            if analysis:
                offsets = embeddings[:, 1:3]
                print('offsets:')
                print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), offsets[:, 0]))))
                print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), offsets[:, 1]))))
                dist = get_distances(embeddings)
                print('dist:\n', dist)
                utl.show_elbow_curve(embeddings)

            utl.decode_clustered_embeddings(decoder_model, embeddings, 1, True)

        if predict_multiple:

            n = 10
            n_clusters = 2
            fig_rows = n_clusters + 1
            fig_cols = n

            plt.figure(figsize=(fig_rows * fig_cols, 4))
            indexes = rand.sample(range(1, x.shape[0]), n)

            for i in range(n):
                sample = x[indexes[i], :, :, 0]
                embeddings = utl.get_embeddings(encoder_model, sample, dim=dim, show=False)
                images = utl.decode_clustered_embeddings(decoder_model, embeddings, n_clusters, dim=dim, show=False)

                # display original
                ax = plt.subplot(fig_rows, n, i + 1)
                plt.imshow(sample)
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display cluster reconstructions
                for cluster in range(n_clusters):
                    ax = plt.subplot(fig_rows, n, i + 1 + (cluster+1) * n)
                    img_reconstruct = images[cluster]
                    plt.imshow(img_reconstruct)
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

            plt.show()


if __name__ == '__main__':
    main()
    print('end')
