from typing import Dict, Tuple
import torchvision
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from .models.simple_conv import SimpleConv, SeqConv
from .models.basemodel import ProtoModel
from .preprocesses import PREPROCESSES


MODELS = {
    "seqconv": SeqConv,
    "simple": SimpleConv
}


class Trainer:
    def __init__(self, dev_dir, test_dir, train_dir, preprocess: str, model: str, batch_size: int = 4):
        self.pre_process = PREPROCESSES[preprocess]

        self.trainset = torchvision.datasets.ImageFolder(train_dir, transform=self.pre_process)
        self.testset = torchvision.datasets.ImageFolder(test_dir, transform=self.pre_process)
        self.devset = torchvision.datasets.ImageFolder(dev_dir, transform=self.pre_process)

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.devloader = torch.utils.data.DataLoader(
            self.devset, batch_size=batch_size, shuffle=True, num_workers=2)

        print(f"Train set: {len(self.trainset)}")
        print(f"Dev set: {len(self.devset)}")
        print(f"Test set: {len(self.testset)}")

        self.classes: Dict[int, str] = self.trainset.class_to_idx
        dev_classes = self.devset.class_to_idx
        test_classes = self.testset.class_to_idx

        assert self.classes == test_classes, "Classes from train and test differ"
        assert self.classes == dev_classes, "Classes from train and test differ"

        self.model: ProtoModel = MODELS[model](len(self.classes))

    def train(
            self,
            n_epochs: int = 100,
            learning_rate: float = 0.001,
            momentum: float = 0.5,
            log_interval: int = 100,
            optimizer: str = "SGD",
            min_lr: float = 1.0000e-08
    ):
        if optimizer == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        train_losses = []
        train_counter = []
        test_losses = []

        best = 0
        for epoch in range(1, n_epochs + 1):
            dev_accuracy = self._epoch(
                epoch=epoch,
                optimizer=optimizer,
                log_interval=log_interval,
                lr_scheduler=lr_scheduler,

                train_losses=train_losses,
                train_counter=train_counter
            )
            if dev_accuracy > best:
                print(f"Saving best model... {dev_accuracy:.04f}")
                best = dev_accuracy
                torch.save(self.model.state_dict(), './results/model.pth')
                torch.save(optimizer.state_dict(), './results/optimizer.pth')

            if optimizer.param_groups[0]['lr'] < min_lr:
                print("Interrupting, LR too small")
                break

        self.model.load_state_dict(
            torch.load('./results/model.pth')
        )
        test_loss, accuracy = self.eval(
            dataset_loader=self.testloader,
            losses_list=test_losses
        )
        self._print_accuracy("Test set", test_loss, accuracy, self.testloader.dataset)
        return train_losses, test_losses

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

        dev_loss, dev_correct = self.eval(dataset_loader=self.devloader)
        self._print_accuracy("Dev", dev_loss, dev_correct, self.devloader.dataset)
        lr_scheduler.step(dev_loss)
        return int(dev_correct) / len(self.devloader.dataset)

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
        return test_loss, int(correct)

    @staticmethod
    def _print_accuracy(dataset_name, loss, correct, dataset):
        print(
            f"\n{dataset_name}: Avg. loss: {loss:.4f}, "
            f"Accuracy: {correct}/{len(dataset)} ({100. * correct / len(dataset):.0f}%)\n"
        )
