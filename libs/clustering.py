import numpy as np
import libs.utilities as utl
import matplotlib.pyplot as plt
from libs.concepts import Concept


np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


def clear_diagonal(data):
    assert len(data.shape) == 2
    assert data.shape[0] == data.shape[1]
    indices = range(data.shape[0])
    data[indices, indices] = 0


def show_stats(data, show_data=False, name='-'):
    print('----------------')
    print('showing stats for', name, 'with shape', data.shape)
    print('\tmax ', np.max(data))
    print('\tmin ', np.min(data))
    print('\tsum ', np.sum(data))
    print('\tmean', np.mean(data))
    print('\tstd ', np.std(data))
    print('\tvar ', np.var(data))
    if show_data:
        print('\tdata\n', data)
    print('----------------')


def clustering_iteration_debug(show=False):
    clustering_model = Concept.BEZIER.get_model_clustering()

    similarity_matrix = np.load('574_cluster_matrix.npy') + .4
    embeddings = np.load('574_embeddings.npy') + .4
    clear_diagonal(similarity_matrix)
    show_stats(similarity_matrix, True, 'similarity_matrix')
    show_stats(embeddings, False, 'embeddings')

    if show:
        plt.imshow(similarity_matrix)
        plt.show()

    row_sums = np.sum(similarity_matrix, axis=0)
    show_stats(row_sums, True, 'row_sums')
    max_row_index = np.argmax(row_sums, axis=0)
    print('max_row_index', max_row_index)
    max_row = similarity_matrix[max_row_index]
    show_stats(max_row, True, 'max_row')
    max_col_index = np.argmax(max_row, axis=0)
    print('max_col_index', max_col_index)
    print('max_element', similarity_matrix[max_row_index, max_col_index])
    print('max_element', similarity_matrix[[max_row_index], [max_col_index]])

    weight_1 = 1
    weight_2 = 1
    cluster_1 = embeddings[max_row_index]
    cluster_2 = embeddings[max_col_index]
    merged_cluster = (cluster_1 * weight_1 + cluster_2 * weight_2) / (weight_1 + weight_2)
    print(cluster_1)
    print(cluster_2)
    print(merged_cluster)
    print(merged_cluster - cluster_1)
    embeddings[max_row_index] = merged_cluster
    embeddings = np.delete(embeddings, (max_col_index, ), axis=0)
    show_stats(embeddings, False, 'embeddings')

    similarity_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
    show_stats(similarity_matrix, False, 'similarity_matrix')


def clustering_iteration(embeddings, clusters, similarity_matrix, strategy, show=False):

    if show:
        plt.imshow(similarity_matrix)
        plt.savefig('similarity_matrix_{}.png'.format(len(embeddings)))

    if strategy == 1:
        row_sums = np.sum(similarity_matrix, axis=0)
        cluster_1_index = np.argmax(row_sums, axis=0)
        max_row = similarity_matrix[cluster_1_index]
        cluster_2_index = np.argmax(max_row, axis=0)

    elif strategy == 2:
        indices = np.where(similarity_matrix == similarity_matrix.max())
        cluster_1_index = indices[0][0]
        cluster_2_index = indices[1][0]

    else:
        return

    cluster_1 = clusters[cluster_1_index]
    cluster_2 = clusters[cluster_2_index]
    clusters[cluster_1_index] = cluster_1 + cluster_2
    del clusters[cluster_2_index]

    if show:
        print('merging {} and {} with similarity {}:\n\t{}\n\t{}'.format(
            cluster_1_index, cluster_2_index,
            similarity_matrix.max(),
            cluster_1, cluster_2))

    [embedding_1, embedding_2] = embeddings[[cluster_1_index, cluster_2_index]]
    embedding = (embedding_1 * len(cluster_1) + embedding_2 * len(cluster_2)) / (len(cluster_1) + len(cluster_2))
    embeddings[cluster_1_index] = embedding
    embeddings = np.delete(embeddings, (cluster_2_index,), axis=0)

    return embeddings, clusters


def generate_image(embeddings, decoder_model):
    images = []
    for embedding in embeddings:
        encoding, center = utl.extract_encoding_and_center(embedding)
        image = utl.gen_image(decoder_model, encoding, center, show=False)
        images.append(image)

    image = np.amax(images, axis=0)
    plt.imshow(image)
    plt.savefig('clustered_image_{}.png'.format(len(embeddings)))


def cluster(embeddings, concept, similarity_threshold, show=False):

    _, _, decoder_model = concept.get_model_autoencoder()
    clustering_model = concept.get_model_clustering()
    clusters = [[i] for i in range(len(embeddings))]

    while len(embeddings) > 1:
        similarity_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings) + 0.4
        clear_diagonal(similarity_matrix)

        if similarity_matrix.max() < similarity_threshold:
            break

        embeddings, clusters = clustering_iteration(
            embeddings, clusters, similarity_matrix, strategy=2, show=show)

        if show:
            print('\tcurrent clusters:\n\t\t' + '\n\t\t'.join([str(it) for it in clusters if len(it) > 1]))
            if 0 < len(embeddings) <= 10:
                generate_image(
                    [embeddings[i] for i in range(len(embeddings)) if len(clusters[i]) > 3],
                    decoder_model)

    return embeddings, clusters


def main():
    embeddings = np.load('../generator/data/574_embeddings.npy')
    cluster(embeddings, Concept.BEZIER, similarity_threshold=0.8, show=False)


if __name__ == '__main__':
    main()
