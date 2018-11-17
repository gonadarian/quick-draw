import numpy as np
import models as mdls
import random as rand
import utilities as utl
import matplotlib.pyplot as plt
from scipy import spatial
from keras.callbacks import TensorBoard, ModelCheckpoint


preload = False
train = True
predict = False
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


# TODO: move to utilities
def get_embeddings(x, idx, show=False):
    # prepare sample
    assert x.shape[1:] == (28, 28, 1)
    x_sample = x[[idx], ...]
    assert x_sample.shape == (1, 28, 28, 1)

    # get prediction
    y_pred = encoder_model.predict(x_sample)
    assert y_pred.shape == (1, 28, 28, 17)
    y_pred = y_pred[0]

    # extract relevant pixels
    x_sample = x_sample.reshape(28, 28)
    idx_sample = np.argwhere(x_sample == 1)
    assert idx_sample.shape[1:] == (2, )
    embeddings = y_pred[idx_sample[:, 0], idx_sample[:, 1], :]

    # convert from relative positions which can't be compared,
    # to absolute ones suitable for k-means clustering
    centers = idx_sample[:,[1,0]]/27-.5 + embeddings[:,1:3]
    embeddings[:,1:3] = centers

    if show:
        plt.imshow(x_sample)
        plt.show()

    return embeddings, x_sample


def load_data():
    x = np.load('generator\data\line_mixed_samples_v2_7234x28x28x1.npy')
    assert x.shape[1:] == (28, 28, 1)
    m = x.shape[0]
    x = np.reshape(x, (m, 28, 28, 1))
    y = np.load('generator\data\line_mixed_encodings_v2_7234x28x28x17.npy')
    assert y.shape[1:] == (28, 28, 17)
    assert y.shape[0] == m
    return x, y, m


x, y, m = load_data()


if preload:
    encoder_model = mdls.load_encoder_model()

else:
    encoder_model = mdls.create_encoder_model()
    encoder_model.compile(optimizer='adam', loss='mean_squared_error')


if train:
    encoder_model.fit(
        x, y,
        epochs=60,
        batch_size=32,
        shuffle=True,
        validation_data=(x, y),
        callbacks=[
            TensorBoard(log_dir='C:\Logs'),
            ModelCheckpoint(
                'models\lines_encoded\lines_mixed_encoded_v2-{epoch:03d}-{val_loss:.6f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ]
    )


if predict:

    predict_single = False
    predict_multiple = True

    decoder_model = mdls.load_decoder_model()

    if predict_single:

        idx = rand.randint(0, x.shape[0])
        print('random index:', idx)
        embeddings, _ = get_embeddings(x, idx, True)

        if analysis:
            offsets = embeddings[:, 1:3]
            print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), offsets[:, 0]))))
            print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), offsets[:, 1]))))
            dist = get_distances(embeddings)
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
            index = indexes[i]
            embeddings, sample = get_embeddings(x, index, False)
            images = utl.decode_clustered_embeddings(decoder_model, embeddings, n_clusters, False)

            # display original
            ax = plt.subplot(fig_rows, n, i + 1)
            img_original = sample  # .reshape(28, 28)
            plt.imshow(img_original)
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


print('end')
