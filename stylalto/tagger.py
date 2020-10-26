from typing import Dict, Tuple, Optional, Union, List, Any
from .trainer import MODELS
import json
from .models.basemodel import ProtoModel
from .preprocesses import PREPROCESSES
from .datasets.utils import ReuseClassesImageFolder
from .datasets.extractor import read_alto_for_tagging
import tqdm


class Tagger:
    def __init__(self,
                 preprocess: str,
                 model: str,
                 class_to_idx: Dict[str, int],
                 device: str = "cpu"):

        self.pre_process = PREPROCESSES[preprocess]
        self.classes: Dict[str, int] = class_to_idx
        self.idx_to_classes: Dict[int, str] = {idx: label for label, idx in self.classes.items()}
        self.model: ProtoModel = MODELS[model](len(self.classes))

        self.use_cuda = device.startswith("cuda")
        self.device = device
        if self.use_cuda:
            self.model.cuda(device)

    @classmethod
    def load_from_prefix(cls, prefix_path) -> "Tagger":
        with open(f"{prefix_path}_config.json") as f:
            conf = json.load(f)
            o = cls(preprocess=conf["preprocess"], model=conf["model"], class_to_idx=conf["class_to_idx"])
        o.model.load_from_path(f"{prefix_path}.pth")
        return o

    def tag(self, xmls: List[str], batch_size: int = 4):
        for xml_path in tqdm.tqdm(xmls):
            bboxes, image, xml = read_alto_for_tagging(xml_path)
