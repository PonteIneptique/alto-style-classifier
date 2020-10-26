from torch.nn import Module
import torch
from typing import Tuple, List


class ProtoModel(Module):
    def __init__(self, classes: int):
        super(ProtoModel, self).__init__()
        self.classes = classes

    def predict(self, batch):
        raise NotImplementedError("This function was not implemented")

    def get_loss_object(self, output, target, **kwargs):
        """ Returns a loss object """
        raise NotImplementedError("This function was not implemented")

    def predict_on_forward(self, output):
        return output.data.max(1, keepdim=True)[1]

    def load_from_path(self, path):
        self.load_state_dict(torch.load(path))
