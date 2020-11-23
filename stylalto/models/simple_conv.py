import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .basemodel import ProtoModel


class SimpleConv(ProtoModel):
    def __init__(self, classes: int):
        super(SimpleConv, self).__init__(classes)

        self.conv1 = nn.Conv2d(1, out_channels=10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 40)
        self.fc2 = nn.Linear(40, classes)

        # https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
        # output_width=((W-F+2*P )/S)+1

    def forward(self, x):
        # torch.Size([batch_size, 1, width, height])
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        # torch.Size([batch_size, out_channels, (((28+(padding*2)-1*(kernel_size-1)-1)/2)+1, 12])
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # torch.Size([batch_size, out_channels_2, 4, 4])
        x = x.view(-1, 320)
        # torch.Size([batch_size, 320])
        x = F.relu(self.fc1(x))
        # torch.Size([batch_size, linear_1_out])
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # torch.Size([batch_size, n_classes])
        return x

    def get_loss_object(self, output, target, **kwargs):
        return F.cross_entropy(output, target, **kwargs)


class SeqConv(ProtoModel):
    # https://www.analyticsvidhya.com/blog/2020/07/how-to-train-an-image-classification-model-in-pytorch-and-tensorflow/
    def __init__(self, classes):
        super(SeqConv, self).__init__(classes)

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, self.classes)
        )

        self.softmax = nn.Softmax()

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def get_loss_object(self, output, target, **kwargs):
        return F.cross_entropy(output, target, **kwargs)

    def predict(self, batch) -> Tuple[List[int], List[float]]:
        x = self.forward(batch)
        confidence, preds = self.softmax(x).max(1)
        return preds.tolist(), confidence.tolist()
