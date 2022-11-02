import numpy as np
from torch import nn
from torchvision import transforms
import torch
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity
from src.models.cars_model import CarsModel


class VerificationCarsModel:
    """ VerificationCarsModel class
    Attributes:
        cars_model_parameters: CarsModel parameters
        gpus: list of GPU device numbers
        emb_size: embedding output size
        weights: path fow model weights (if None use ' ')
        """

    def __init__(self, cars_model_parameters: dict, gpus: list, emb_size: int,
                 weights: str):

        self.cars_model_parameters = cars_model_parameters
        self.gpus = gpus
        self.emb_size = emb_size
        self.weights = weights

        self.load_model()

    def preprocess(self, img: Image, resize: int = 512):
        torch_img = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((resize, resize))])(
            img)

        return torch_img.view(1, 3, resize, resize).to(self.device)

    def load_model(self):
        """ Load model  """
        if self.weights != '':
            self.model = torch.load(self.weights, map_location='cpu')
        else:
            self.model = CarsModel(**self.cars_model_parameters)

        if len(self.gpus) == 1:
            self.device = f'cuda:{self.gpus[0]}'
            self.model = self.model.to(self.device)
        else:
            self.device = 'cuda'
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
            self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, img_1: Image, img_2: Image, resize: int = 512):
        self.model.eval()
        img_1 = self.preprocess(img_1, resize)
        img_2 = self.preprocess(img_2, resize)

        emb_1 = self.model(img_1).detach().cpu().numpy()
        emb_2 = self.model(img_2).detach().cpu().numpy()

        return cosine_similarity(emb_1, emb_2)[0][0]


def get_model(model_name: str, model_parameters: dict):
    """ Get model function
    Arguments:
        model_name: model name
        model_parameters: model parameters
    """
    model_mapping = {'cars_model': VerificationCarsModel(**model_parameters)}

    return model_mapping[model_name]
