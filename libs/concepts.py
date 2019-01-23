from enum import Enum
import libs.models as mdls
import libs.datasets as ds


class Concept(Enum):

    LINE = (
        'line',
        1.,
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
        1.,
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

    BEZIER = (
        'bezier',
        1.,
        mdls.create_autoencoder_model,
        mdls.load_autoencoder_bezier_model,
        mdls.create_matrix_encoder_model,
        mdls.load_matrix_encoder_bezier_model,
        mdls.create_clustering_model,
        mdls.load_clustering_bezier_model,
        ds.load_images_bezier_centered,
        ds.load_images_bezier_shifted,
        ds.load_images_bezier_mixed,
        ds.load_images_bezier_clustered,
    )

    STAR = (
        'star',
        1.,
        mdls.create_autoencoder_model,
        mdls.load_autoencoder_star_model,
        mdls.create_matrix_encoder_model,
        mdls.load_matrix_encoder_star_model,
        mdls.create_clustering_model,
        mdls.load_clustering_star_model,
        ds.load_images_star_centered,
        ds.load_images_star_shifted,
        ds.load_images_star_mixed,
        ds.load_images_star_clustered,
    )

    SQUARE = (
        'square',
        1.,
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

    def __init__(self, code, sample_threshold,
                 model_autoencoder_creator, model_autoencoder,
                 model_matrix_encoder_creator, model_matrix_encoder,
                 model_clustering_creator, model_clustering,
                 dataset_centered, dataset_shifted, dataset_mixed, dataset_clustered):

        self.code = code
        self.sample_threshold = sample_threshold
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

        self._model_autoencoder = None
        self._model_matrix_encoder = None
        self._model_clustering = None
        self._dataset_centered = None
        self._dataset_shifted = None
        self._dataset_mixed = None
        self._dataset_clustered = None

    def get_model_autoencoder(self, trained=True):
        self._model_autoencoder = (self.model_autoencoder_creator() if not trained else
                                   self.model_autoencoder() if not self._model_autoencoder else
                                   self._model_autoencoder)

        return self._model_autoencoder

    def get_model_matrix_encoder(self, trained=True):
        self._model_matrix_encoder = (self.model_matrix_encoder_creator() if not trained else
                                      self.model_matrix_encoder() if not self._model_matrix_encoder else
                                      self._model_matrix_encoder)

        return self._model_matrix_encoder

    def get_model_clustering(self, trained=True):
        self._model_clustering = (self.model_clustering_creator() if not trained else
                                  self.model_clustering() if not self._model_clustering else
                                  self._model_clustering)

        return self._model_clustering

    def get_dataset_centered(self):
        self._dataset_centered = (self.dataset_centered() if not self._dataset_centered else
                                  self._dataset_centered)

        return self._dataset_centered

    def get_dataset_shifted(self):
        self._dataset_shifted = (self.dataset_shifted() if not self._dataset_shifted else
                                 self._dataset_shifted)

        return self._dataset_shifted

    def get_dataset_mixed(self):
        self._dataset_mixed = (self.dataset_mixed() if not self._dataset_mixed else
                               self._dataset_mixed)

        return self._dataset_mixed

    def get_dataset_clustered(self):
        self._dataset_clustered = (self.dataset_clustered() if not self._dataset_clustered else
                                   self._dataset_clustered)

        return self._dataset_clustered
