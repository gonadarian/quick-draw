import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


preload = True
train = False
predict = True
analysis = False


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def gen_decoder_model(autoencoder, show=False):
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
    decoder = Model(inputs=input, outputs=x)

    if show:
        decoder.summary()

    return decoder


def gen_image(decoder, encoding, center, show=False):
    assert encoding.shape == (14, )
    import matplotlib.pyplot as plt

    # this is a 14-number encoding for one of the lines in the test set
    encoding = np.reshape(encoding, (1, 1, 1, 14))
    image = decoder.predict(encoding)
    assert image.shape == (1, 28, 28, 1)
    image = image.reshape(28, 28)

    from_row = 28 + center[1]
    from_col = 28 + center[0]
    shifted = np.zeros((84, 84))
    shifted[from_row:from_row+28, from_col:from_col+28] = image
    shifted = shifted[28:56, 28:56]
    assert shifted.shape == (28, 28)

    if show:
        plt.gray()
        plt.imshow(shifted)
        plt.show()

    return shifted


def get_model():
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


def get_distances(encodings):
    count = encodings.shape[0]
    distances = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            distances[i, j] = spatial.distance.cosine(encodings[i, :], encodings[j, :])

    for idx in range(count):
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), distances[idx, :]))), '\n')

    return distances


def show_elbow_curve(encodings, show=False):
    count = encodings.shape[0]
    distance_list = list()
    n_cluster = range(1, min(count - 1, 20))

    for n in n_cluster:
        kmeans = KMeans(n_clusters=n, random_state=0).fit(encodings)
        print(kmeans.labels_)
        distance = np.average(np.min(cdist(encodings, kmeans.cluster_centers_, 'euclidean'), axis=1))
        print(distance)
        distance_list.append(distance)

    if show:
        plt.plot(n_cluster, distance_list)
        plt.title('elbow curve')
        plt.show()


def get_embeddings(x, idx, show=False):
    # prepare sample
    assert x.shape[1:] == (28, 28, 1)
    x_sample = x[[idx], ...]
    assert x_sample.shape == (1, 28, 28, 1)

    # get prediction
    y_pred = encoder.predict(x_sample)
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


def decode_clustered_embeddings(decoder, embeddings, n_clusters=1, show=False):
    assert embeddings.shape[1:] == (17, )

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    images = []

    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i][1:3]
        center = np.rint(center * 27).astype(int)
        encoding = kmeans.cluster_centers_[i][3:17]
        print('cluster center:', center)
        print('cluster encoding:', encoding)
        image = gen_image(decoder, encoding, center, show)
        images.append(image)

    return images


def load_data():
    # x = np.load('generator\line_samples_v2_7234x28x28x1.npy')
    x = np.load('generator\line_mixed_samples_v2_7234x28x28x1.npy')
    assert x.shape[1:] == (28, 28, 1)
    m = x.shape[0]
    x = np.reshape(x, (m, 28, 28, 1))
    # y = np.load('generator\line_encodings_v2_7234x28x28x16.npy')
    y = np.load('generator\line_mixed_encodings_v2_7234x28x28x17.npy')
    assert y.shape[1:] == (28, 28, 17)
    assert y.shape[0] == m
    return (x, y, m)


x, y, m = load_data()


if preload:
    from keras.models import load_model
    # encoder = load_model('models\lines_encoded\lines_encoded_v2-047-0.000024.hdf5')
    encoder = load_model('models\lines_encoded\lines_mixed_encoded_v2-091-0.000072.hdf5')
else:
    encoder = get_model()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    encoder.compile(optimizer='adam', loss='mean_squared_error')


if train:
    from keras.callbacks import TensorBoard, ModelCheckpoint
    encoder.fit(
        x, y,
        epochs=60,
        batch_size=32,
        shuffle=True,
        # validation_split=0.1,
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

    import matplotlib.pyplot as plt
    import random

    autoencoder = load_model('models\lines\lines_autoencoder_v2-385-0.0047.hdf5')
    decoder = gen_decoder_model(autoencoder)

    if predict_single:

        idx = random.randint(0, x.shape[0])
        # idx = 338  # 149
        print('random index:', idx)
        embeddings, _ = get_embeddings(x, idx, True)

        if analysis:
            offsets = embeddings[:, 1:3]
            print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), offsets[:, 0]))))
            print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), offsets[:, 1]))))
            dist = get_distances(embeddings)
            show_elbow_curve(embeddings)

        decode_clustered_embeddings(decoder, embeddings, 1, True)

    if predict_multiple:

        n = 10
        n_clusters = 2
        fig_rows = n_clusters + 1
        fig_cols = n

        plt.figure(figsize=(fig_rows * fig_cols, 4))
        indexes = random.sample(range(1, x.shape[0]), n)

        for i in range(n):
            index = indexes[i]
            embeddings, sample = get_embeddings(x, index, False)
            images = decode_clustered_embeddings(decoder, embeddings, n_clusters, False)

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
