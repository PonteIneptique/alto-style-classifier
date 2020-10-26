from torchvision.datasets.folder import ImageFolder, default_loader
from typing import Optional, Dict


class ReuseClassesImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, class_to_idx: Optional[Dict[str, int]] = None):

        self.class_to_idx, self.classes = None, None
        if class_to_idx:
            self.class_to_idx = class_to_idx
            self.classes = sorted(list(self.class_to_idx.keys()))

        super(ReuseClassesImageFolder, self).__init__(
            root=root, loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

    def _find_classes(self, dir):
        if self.class_to_idx:
            return self.classes, self.class_to_idx
        return super(ReuseClassesImageFolder, self)._find_classes(dir)