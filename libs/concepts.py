from enum import Enum
import libs.models as mdls
import libs.datasets as ds


class Concept(Enum):

    LINE = (
        'line',
        mdls.create_autoencoder_model,
        mdls.load_autoencoder_line_model,
        mdls.create_matrix_encoder_model,
        mdls.load_matrix_encoder_line_model,
        mdls.create_clustering_model,
        mdls.load_clustering_line_model,
        ds.load_images_line_centered,
        ds.load_images_line_shifted,
        ds.load_images_line_mixed,
        ds.load_images_line_clustered,
    )

    ELLIPSE = (
        'ellipse',
        mdls.create_autoencoder_model,
        mdls.load_autoencoder_ellipse_model,
        mdls.create_matrix_encoder_model,
        mdls.load_matrix_encoder_ellipse_model,
        mdls.create_clustering_model,
        mdls.load_clustering_ellipse_model,
        ds.load_images_ellipse_centered,
        ds.load_images_ellipse_shifted,
        ds.load_images_ellipse_mixed,
        ds.load_images_ellipse_clustered,
    )

    SQUARE = (
        'square',
        mdls.create_graph_autoencoder_model,
        mdls.load_graph_autoencoder_model,
        None,
        None,
        None,
        None,
        ds.load_graphs_square_centered,
        None,
        None,
        None,
    )

    def __init__(self, code,
                 model_autoencoder_creator, model_autoencoder,
                 model_matrix_encoder_creator, model_matrix_encoder,
                 model_clustering_creator, model_clustering,
                 dataset_centered, dataset_shifted, dataset_mixed, dataset_clustered):

        self.code = code
        self.model_autoencoder_creator = model_autoencoder_creator
        self.model_autoencoder = model_autoencoder
        self.model_matrix_encoder_creator = model_matrix_encoder_creator
        self.model_matrix_encoder = model_matrix_encoder
        self.model_clustering_creator = model_clustering_creator
        self.model_clustering = model_clustering
        self.dataset_centered = dataset_centered
        self.dataset_shifted = dataset_shifted
        self.dataset_mixed = dataset_mixed
        self.dataset_clustered = dataset_clustered
