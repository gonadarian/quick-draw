from enum import Enum
import libs.models as mdls
import libs.datasets as ds


class Concept(Enum):

    LINE = (
        'line',
        mdls.create_autoencoder_model_27x27,
        mdls.load_autoencoder_line_model_27x27,
        mdls.create_encoder_model_27x27,
        mdls.load_matrix_encoder_line_model_27x27,
        ds.load_images_line_27x27_centered,
        ds.load_images_line_27x27_shifted,
        ds.load_images_line_27x27_mixed,
    )

    ELLIPSE = (
        'ellipse',
        mdls.create_autoencoder_model_27x27,
        mdls.load_autoencoder_ellipse_model_27x27,
        mdls.create_encoder_model_27x27,
        mdls.load_matrix_encoder_ellipse_model_27x27,
        ds.load_images_ellipse_27x27_centered,
        ds.load_images_ellipse_27x27_shifted,
        ds.load_images_ellipse_27x27_mixed,
    )

    def __init__(self, code,
                 model_autoencoder_creator, model_autoencoder,
                 model_matrix_encoder_creator, model_matrix_encoder,
                 dataset_centered, dataset_shifted, dataset_mixed):
        self.code = code
        self.model_autoencoder_creator = model_autoencoder_creator
        self.model_autoencoder = model_autoencoder
        self.model_matrix_encoder_creator = model_matrix_encoder_creator
        self.model_matrix_encoder = model_matrix_encoder
        self.dataset_centered = dataset_centered
        self.dataset_shifted = dataset_shifted
        self.dataset_mixed = dataset_mixed
