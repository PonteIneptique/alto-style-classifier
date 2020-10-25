from typing import Dict, Tuple
import torchvision
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from .models.simple_conv import SimpleConv
from .preprocesses import PREPROCESSES


class Trainer:
    def __init__(self, dev_dir, test_dir, train_dir, preprocess: str):
        self.pre_process = PREPROCESSES[preprocess]

        self.trainset = torchvision.datasets.ImageFolder(train_dir, transform=self.pre_process)
        self.testset = torchvision.datasets.ImageFolder(test_dir, transform=self.pre_process)
        self.devset = torchvision.datasets.ImageFolder(dev_dir, transform=self.pre_process)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=True, num_workers=2)
        self.devloader = torch.utils.data.DataLoader(self.devset, batch_size=4, shuffle=True, num_workers=2)

        print(f"Train set: {len(self.trainset)}")
        print(f"Dev set: {len(self.devset)}")
        print(f"Test set: {len(self.testset)}")

        self.classes: Dict[int, str] = self.trainset.class_to_idx
        dev_classes = self.devset.class_to_idx
        test_classes = self.testset.class_to_idx

        assert self.classes == test_classes, "Classes from train and test differ"
        assert self.classes == dev_classes, "Classes from train and test differ"

        self.model = SimpleConv(len(self.classes))

    def train(
            self,
            n_epochs: int = 100,
            learning_rate: float = 0.001,
            momentum: float = 0.5,
            log_interval: int = 100
    ):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        train_losses = []
        train_counter = []
        test_losses = []

        for epoch in range(1, n_epochs + 1):
            self._epoch(
                epoch=epoch,
                optimizer=optimizer,
                log_interval=log_interval,
                lr_scheduler=lr_scheduler,

                train_losses=train_losses,
                train_counter=train_counter
            )
            test_loss, accuracy = self.eval(
                dataset_loader=self.testloader,
                losses_list=test_losses
            )
            self._print_accuracy("Test set", test_loss, accuracy, self.testloader.dataset)

    def _epoch(self, epoch, optimizer, log_interval, train_losses, train_counter, lr_scheduler):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.trainloader.dataset),
                           100. * batch_idx / len(self.trainloader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(self.trainloader.dataset)))
                torch.save(self.model.state_dict(), './results/model.pth')
                torch.save(optimizer.state_dict(), './results/optimizer.pth')

        dev_loss, dev_correct = self.eval(dataset_loader=self.devloader)
        self._print_accuracy("Dev", dev_loss, dev_correct, self.devloader.dataset)
        lr_scheduler.step(dev_loss)

    def eval(self, dataset_loader, losses_list=None) -> Tuple[float, float]:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataset_loader:
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(dataset_loader.dataset)
        if isinstance(losses_list, list):
            losses_list.append(test_loss)
        return test_loss, correct

    @staticmethod
    def _print_accuracy(dataset_name, loss, correct, dataset):
        print(
            f"\n{dataset_name}: Avg. loss: {loss:.4f}, "
            f"Accuracy: {correct}/{len(dataset)} ({100. * correct / len(dataset):.0f}%)\n"
        )
