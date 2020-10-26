from typing import Dict, Tuple, Optional, Union, List
import torch.utils.data
import torch.optim as optim
from sklearn.metrics import classification_report

from .models.simple_conv import SimpleConv, SeqConv
from .models.basemodel import ProtoModel
from .preprocesses import PREPROCESSES
from .datasets.utils import ReuseClassesImageFolder


MODELS = {
    "seqconv": SeqConv,
    "simple": SimpleConv
}


class Trainer:
    def __init__(self, nb_classes: int, preprocess: str, model: str, batch_size: int = 4,
                 device: str = "cpu", class_to_idx: Optional[Dict[str, int]] = None):

        self.pre_process = PREPROCESSES[preprocess]
        self.classes: Dict[str, int] = class_to_idx
        self.batch_size: int = batch_size

        self.model: ProtoModel = MODELS[model](nb_classes)

        self.use_cuda = device.startswith("cuda")
        self.device = device
        if self.use_cuda:
            self.model.cuda(device)

    @property
    def idx_to_classes(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.classes.items()}

    def generate_dataset(self, path):
        # Load train
        dataset = ReuseClassesImageFolder(path, transform=self.pre_process, class_to_idx=self.classes)
        # If the classes are not hardcoded, set-up the information
        if not self.classes:
            self.classes = dataset.class_to_idx
        # Get the data loader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        return dataset, loader

    def train(
            self,
            train_dir,
            dev_dir,
            n_epochs: int = 100,
            learning_rate: float = 0.001,
            momentum: float = 0.5,
            log_interval: int = 100,
            optimizer: str = "SGD",
            min_lr: float = 1.0000e-08,
            patience: int = 5
    ):
        trainset, trainloader = self.generate_dataset(train_dir)
        devset, devloader = self.generate_dataset(dev_dir)

        print(f"Train set: {len(trainset)}")
        print(f"Dev set: {len(devset)}")

        if optimizer == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True,
            patience=patience
        )
        train_losses = []
        train_counter = []
        test_losses = []

        best = 0

        try:
            for epoch in range(1, n_epochs + 1):
                dev_accuracy = self._epoch(
                    trainloader=trainloader,
                    devloader=devloader,
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
        except KeyboardInterrupt:
            print(f"\nKeyboard interrupt: loading best model at {best:.2f}\n")

        self.model.load_state_dict(
            torch.load('./results/model.pth')
        )
        return self.model

    def _epoch(self,
               trainloader, devloader,
               epoch, optimizer, log_interval, train_losses, train_counter, lr_scheduler):
        self.model.train()
        for batch_idx, (data, target) in enumerate(trainloader):

            if self.use_cuda:
                data, target = data.cuda(self.device), target.cuda(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = self.model.get_loss_object(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f'Train Epoch: {epoch} '
                    f'[{str(batch_idx * len(data)).zfill(len(str(len(trainloader.dataset))))}/'
                    f'{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(trainloader.dataset)))

        dev_loss, dev_correct = self.eval(dataset_loader=devloader)
        self._print_accuracy("Dev", dev_loss, dev_correct, devloader.dataset)
        lr_scheduler.step(dev_loss)
        return int(dev_correct) / len(devloader.dataset)

    def eval(self, dataset_loader, return_preds_and_truth: bool = False) -> Union[
                                                                                Tuple[float, float],
                                                                                Tuple[
                                                                                    float, float, List[int], List[int]
                                                                                ],
                                                                            ]:
        self.model.eval()
        test_loss = 0
        correct = 0
        truthes, preds = [], []
        with torch.no_grad():
            for data, target in dataset_loader:

                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.model.get_loss_object(output, target, size_average=False).item()
                pred = self.model.predict_on_forward(output)
                correct += pred.eq(target.data.view_as(pred)).sum()
                if return_preds_and_truth:
                    truthes.extend(target.tolist())
                    preds.extend(pred.tolist())

        test_loss /= len(dataset_loader.dataset)

        if return_preds_and_truth:
            return test_loss, int(correct), truthes, preds

        return test_loss, int(correct)

    def get_eval_details(self, gts, preds):
        print(
            classification_report(
                y_true=gts, y_pred=preds,
                target_names=[self.idx_to_classes[idx] for idx, _ in enumerate(self.classes)]
            )
        )

    @staticmethod
    def _print_accuracy(dataset_name, loss, correct, dataset):
        print(
            f"\n{dataset_name}: Avg. loss: {loss:.4f}, "
            f"Accuracy: {correct}/{len(dataset)} ({100. * correct / len(dataset):.0f}%)\n"
        )
