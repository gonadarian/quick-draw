from enum import Enum
import libs.models as mdls
import libs.datasets as ds


class Concept(Enum):

    LINE = (
        'line',
        mdls.get_model_autoencoder_27x27,
        mdls.load_autoencoder_model_27x27,
        ds.load_images_line_27x27_centered
    )

    ELLIPSE = (
        'ellipse',
        mdls.get_model_autoencoder_27x27,
        None,
        None,
    )

    def __init__(self, code, model_creator, model_loader, dataset_loader):
        self.code = code
        self.model_creator = model_creator
        self.model_loader = model_loader
        self.dataset_loader = dataset_loader
