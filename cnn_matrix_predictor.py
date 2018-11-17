import numpy as np
import models as mdls
import utilities as utl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


analysis = False


def load_data(show=False, shape_choice=1):
    im = Image.new("L", (28, 28), "black")
    draw = ImageDraw.Draw(im)

    if shape_choice == 0:  # shape: square
        draw.line((5,  5, 5,  22), fill=255)
        draw.line((5,  5, 22, 5), fill=255)
        draw.line((22, 5, 22, 22), fill=255)
        draw.line((5, 22, 22, 22), fill=255)

    if shape_choice == 1:  # shape: rhomboid
        draw.line((5,  14, 14, 5), fill=255)
        draw.line((14, 5,  23, 14), fill=255)
        draw.line((23, 14, 14, 23), fill=255)
        draw.line((14, 23, 5,  14), fill=255)

    if shape_choice == 2:  # shape: tilted, parallel lines
        draw.line((2, 2,  25, 6), fill=255)
        draw.line((2, 8,  25, 12), fill=255)
        draw.line((2, 14, 25, 18), fill=255)
        draw.line((2, 20, 25, 24), fill=255)

    if shape_choice == 3:  # shape: star
        draw.line((5,  10, 23, 18), fill=255)
        draw.line((10, 5,  18, 23), fill=255)
        draw.line((18, 5,  10, 23), fill=255)
        draw.line((23, 10, 5,  18), fill=255)

    data = np.asarray(im)
    if show:
        im.show()

    data = data.astype('float32') / 255.
    return data


def show_clusters(input_image, cluster_images):
    # display regenerated lines alongside original sample
    n_clusters = len(cluster_images)
    fig = plt.figure(figsize=(1, n_clusters + 1))

    for i in range(n_clusters):
        fig.add_subplot(1, n_clusters + 1, i + 1)
        plt.imshow(cluster_images[i])

    fig.add_subplot(1, n_clusters + 1, n_clusters + 1)
    plt.imshow(input_image)
    plt.show()


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

sample = load_data(show=False)
assert sample.shape == (28, 28)

decoder_model = mdls.load_decoder_model()
encoder_model = mdls.load_encoder_model()
clustering_model = mdls.load_clustering_model()

embeddings = utl.get_embeddings(encoder_model, sample, False)
cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
clusters = utl.extract_clusters(cluster_matrix)

if analysis:
    utl.show_elbow_curve(embeddings, True)
    plt.imshow(cluster_matrix)
    plt.show()

images = []
for cluster in clusters:
    if len(cluster) > 2:
        cluster_embeddings = embeddings[list(cluster)]
        cluster_embedding = np.mean(cluster_embeddings, axis=0)
        encoding, center = utl.extract_encoding_and_center(cluster_embedding)
        assert encoding.shape == (14, )
        assert center.shape == (2, )
        image = utl.gen_image(decoder_model, encoding, center, show=False)
        images.append(image)

show_clusters(sample, images)


print('end')
