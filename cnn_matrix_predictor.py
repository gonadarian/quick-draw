import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, load_model
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


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


def show_elbow_curve(encodings, show=False):
    count = encodings.shape[0]
    distance_list = list()
    n_cluster = range(1, min(count - 1, 20))

    for n in n_cluster:
        kmeans = KMeans(n_clusters=n, random_state=0).fit(encodings)
        print('labels:', kmeans.labels_)
        distance = np.average(np.min(cdist(encodings, kmeans.cluster_centers_, 'euclidean'), axis=1))
        print('calculated distance:', distance)
        distance_list.append(distance)

    if show:
        plt.plot(n_cluster, distance_list)
        plt.title('elbow curve')
        plt.show()


def get_embeddings(sample, show=False):
    # prepare sample
    assert sample.shape == (28, 28)
    sample = sample.reshape((1, 28, 28, 1))

    # get prediction
    y_pred = encoder.predict(sample)
    assert y_pred.shape == (1, 28, 28, 17)
    y_pred = y_pred[0]

    # extract relevant pixels
    sample = sample.reshape(28, 28)
    idx_sample = np.argwhere(sample == 1)
    assert idx_sample.shape[1:] == (2, )
    embeddings = y_pred[idx_sample[:, 0], idx_sample[:, 1], :]

    # convert from relative positions which can't be compared,
    # to absolute ones suitable for k-means clustering
    centers = idx_sample[:,[1,0]]/27-.5 + embeddings[:,1:3]
    embeddings[:,1:3] = centers

    if show:
        plt.imshow(sample)
        plt.show()

    return embeddings


def extract_encoding_and_center(cluster):
    center = cluster[1:3]
    center = np.rint(center * 27).astype(int)
    encoding = cluster[3:17]

    return encoding, center


def decode_clustered_embeddings(decoder, embeddings, n_clusters=1, show=False):
    assert embeddings.shape[1:] == (17, )

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    images = []

    for i in range(n_clusters):
        cluster = kmeans.cluster_centers_[i]
        encoding, center = extract_encoding_and_center(cluster)
        print('cluster center:', center)
        print('cluster encoding:', encoding)
        image = gen_image(decoder, encoding, center, show)
        images.append(image)

    return images


def load_data(show=False):
    im = Image.new("L", (28, 28), "black")

    draw = ImageDraw.Draw(im)

    # draw.line((5, 5, 5, 22), fill=255)
    # draw.line((5, 5, 22, 5), fill=255)
    # draw.line((22, 5, 22, 22), fill=255)
    # draw.line((5, 22, 22, 22), fill=255)

    draw.line((5 , 14, 14, 5 ), fill=255)
    draw.line((14, 5 , 23, 14), fill=255)
    draw.line((23, 14, 14, 23), fill=255)
    draw.line((14, 23, 5 , 14), fill=255)

    # draw.line((2 , 2 , 25, 6 ), fill=255)
    # draw.line((2 , 8 , 25, 12), fill=255)
    # draw.line((2 , 14, 25, 18), fill=255)
    # draw.line((2 , 20, 25, 24), fill=255)

    sample = np.asarray(im)
    if show:
        im.show()

    sample = sample.astype('float32') / 255.
    return sample


sample = load_data(False)

# encoder = load_model('models\lines_encoded\lines_encoded_v2-047-0.000024.hdf5')
# encoder = load_model('models\lines_encoded\lines_mixed_encoded_v2-037-0.000102.hdf5')
encoder = load_model('models\lines_encoded\lines_mixed_encoded_v2-091-0.000072.hdf5')
autoencoder = load_model('models\lines\lines_autoencoder_v2-385-0.0047.hdf5')
decoder = gen_decoder_model(autoencoder)

embeddings = get_embeddings(sample, False)
# show_elbow_curve(embeddings, True)

n_clusters = 4
# images = decode_clustered_embeddings(decoder, embeddings, n_clusters, False)

# images = []
# images.append(decode_clustered_embeddings(decoder, embeddings[0:24], 1, False)[0])
# images.append(decode_clustered_embeddings(decoder, embeddings[24:48], 1, False)[0])
# images.append(decode_clustered_embeddings(decoder, embeddings[48:72], 1, False)[0])
# images.append(decode_clustered_embeddings(decoder, embeddings[72:96], 1, False)[0])

images = []
encoding, center = extract_encoding_and_center(embeddings[5])
images.append(gen_image(decoder, encoding, center, False))
encoding, center = extract_encoding_and_center(embeddings[6])
images.append(gen_image(decoder, encoding, center, False))
encoding, center = extract_encoding_and_center(embeddings[30])
images.append(gen_image(decoder, encoding, center, False))
encoding, center = extract_encoding_and_center(embeddings[31])
images.append(gen_image(decoder, encoding, center, False))

fig = plt.figure(figsize=(1, n_clusters + 1))

for i in range(n_clusters):
    fig.add_subplot(1, n_clusters + 1, i + 1)
    plt.imshow(images[i])

fig.add_subplot(1, n_clusters + 1, n_clusters + 1)
plt.imshow(sample)
plt.show()


print('end')
