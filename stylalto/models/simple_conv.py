import torch.nn as nn
import torch.nn.functional as F

from .basemodel import ProtoModel


class SimpleConv(ProtoModel):
    def __init__(self, classes: int):
        super(SimpleConv, self).__init__(classes)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self, x):
        # torch.Size([batch_size, 1, width, height])
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        # torch.Size([batch_size, 10, 12, 12])
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # torch.Size([4, 20, 4, 4])
        x = x.view(-1, 320)
        # torch.Size([4, 320])
        x = F.relu(self.fc1(x))
        # torch.Size([4, 50])
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # torch.Size([4, 3])
        return F.log_softmax(x)

