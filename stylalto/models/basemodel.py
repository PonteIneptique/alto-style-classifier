from torch.nn import Module


class ProtoModel(Module):
    def __init__(self, classes: int):
        super(ProtoModel, self).__init__()
        self.classes = classes

    def predict(self, batch):
        raise NotImplementedError("This function was not implemented")
