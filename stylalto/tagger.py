from typing import Dict, Tuple, Optional, Union, List, Any
from .trainer import MODELS
import json
from .models.basemodel import ProtoModel
from .preprocesses import PREPROCESSES
from .datasets.extractor import read_alto_for_tagging, extract_images_from_bbox_dict_for_tagging, extract_styles, NS
import tqdm
import torch
from lxml.etree import Element, tostring


def get_batches(images, bboxes, batch_size: int, transform):
    batch = []
    for image, bbox in zip(images, bboxes):
        batch.append((transform(image), bbox))
        if len(batch) == batch_size:
            yield torch.stack([img[0] for img in batch], dim=0), [img[1] for img in batch]
            batch = []

    if len(batch):
        yield torch.stack([img[0] for img in batch], dim=0), [img[1] for img in batch]



STYLES_TAG = "{" + NS["a"]+ "}Styles"
TEXTSTYLE_TAG = "{" + NS["a"]+ "}TextStyle"


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
        self.model.eval()

    @classmethod
    def load_from_prefix(cls, prefix_path) -> "Tagger":
        with open(f"{prefix_path}_config.json") as f:
            conf = json.load(f)
            o = cls(preprocess=conf["preprocess"], model=conf["model"], class_to_idx=conf["class_to_idx"])
        o.model.load_from_path(f"{prefix_path}.pth")
        return o

    def check_and_add_styles(self, xml):
        styles = extract_styles(xml)
        reverse_styles = {
            fontstyle.strip(): style_id
            for style_id, fontstyle in styles.items()
        }
        need_to_write = []
        for fontstyle, idx in self.classes.items():
            if fontstyle not in reverse_styles:
                reverse_styles[fontstyle] = f"STYLEALTO_AUTO_{idx}"
                styles[f"STYLEALTO_AUTO_{idx}"] = fontstyle
                need_to_write.append(f"STYLEALTO_AUTO_{idx}")

        style_elem = xml.xpath("//a:Styles", namespaces=NS)
        if len(style_elem) == 0:
            style_elem = Element(STYLES_TAG, nsmap=NS)
            xml.append(style_elem)
        else:
            style_elem = style_elem[0]

        for style_id in need_to_write:
            style_elem.append(
                Element(TEXTSTYLE_TAG, attrib={
                    "FONTSTYLE": styles[style_id].replace("_", ""),
                    "ID": style_id,
                    "FONTSIZE": "1"
                })
            )
        return reverse_styles

    def tag(self, xmls_path: List[str], batch_size: int = 4):
        for xml_path in tqdm.tqdm(xmls_path):
            bboxes, image, xml = read_alto_for_tagging(xml_path)
            label_to_style = self.check_and_add_styles(xml)
            words = {
                word.attrib["ID"]: word
                for word in xml.xpath("//a:String", namespaces=NS)
            }
            images = extract_images_from_bbox_dict_for_tagging(bboxes, image)
            for batch_images, batch_bbox in get_batches(
                    images, bboxes, batch_size=batch_size, transform=self.pre_process):
                preds, confidences = (self.model.predict(batch_images))
                for bbox, pred, confidence in zip(batch_bbox, preds, confidences):
                    words[bbox.id].attrib.update({
                        "STYLEREFS": label_to_style[self.idx_to_classes[pred]],
                        "STYLALTO_CONFIDENCE": f"{confidence:.4f}"
                    })
            with open(f"{xml_path}-predict.xml", "w") as out:
                out.write(tostring(xml, encoding=str))
        return True
