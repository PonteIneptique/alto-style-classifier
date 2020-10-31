from typing import Dict, List, Tuple, Union, Optional
from collections import namedtuple, defaultdict
import lxml.etree as et
import os
import PIL.Image as PILImage
import tqdm
import random
import glob
from dataclasses import dataclass


@dataclass(frozen=True)
class BBOX:
    x1: int
    y1: int
    x2: int
    y2: int
    id: str
    file: Optional[str] = None
    image: Optional[str] = None
    style: Optional[str] = None

    @property
    def coords(self):
        return self.x1, self.y1, self.x2, self.y2


class NoSourceImage(Exception):
    """ Raised when the ALTO is missing a link to an image file """


NS = {"a": "http://www.loc.gov/standards/alto/ns-v2#",
      "stylalto": "https://github.com/PonteIneptique/alto-style-classifier#"}


def temporary_replace_path(xml_path):
    return f"../IMG/{xml_path.replace('.xml', '.jpg')}"


def extract_styles(xml):
    return {
        style.attrib["ID"]: style.attrib["FONTSTYLE"] if style.attrib["FONTSTYLE"] else "_"
        for style in xml.xpath("//a:TextStyle", namespaces=NS)
    }


def read_alto_for_training(alto_xml) -> Tuple[Dict[str, List[BBOX]], str]:
    classes = defaultdict(list)
    xml, image_path = _parse_alto(alto_xml)
    # Retrieve styles
    styles = extract_styles(xml)
    # Retrieve BBOX sorted by style
    for string in xml.xpath("//a:String", namespaces=NS):
        bbox = _compute_bbox_form_node(string, xml_path=alto_xml, image_path=image_path)
        style = styles[string.attrib["STYLEREFS"]]
        classes[style].append(bbox)

    return classes, image_path


def get_image_locally(xml: et.ElementBase, path_xml: str = "", temp_fix: bool = True):
    source_image = xml.xpath("//a:sourceImageInformation/a:fileName/text()", namespaces=NS)
    if not len(source_image):
        raise NoSourceImage(f"{path_xml} is missing the following node"
                            "`/alto/Description/sourceImageInformation/fileName`"
                            "which should contain the path to the image it is about")

    source_image = str(source_image[0])
    if temp_fix:
        source_image = temporary_replace_path(source_image)

    source_image_real_path = os.path.abspath(
        os.path.join(os.path.dirname(path_xml), source_image)
    )
    if not os.path.isfile(source_image_real_path):
        raise NoSourceImage(f"{path_xml} has a wrong path at"
                            "`/alto/Description/sourceImageInformation/fileName`"
                            f": {source_image_real_path}")
    return source_image_real_path


def _parse_alto(alto_xml, temp_fix: bool = False):
    with open(alto_xml) as f:
        xml = et.parse(f)
        source_image_real_path = get_image_locally(xml, path_xml=alto_xml, temp_fix=temp_fix)
    return xml, source_image_real_path


def _compute_bbox_form_node(
        string_elem: et.ElementBase,
        xml_path: Optional[str] = None,
        image_path: Optional[str] = None) -> BBOX:
    x, y, = string_elem.attrib["HPOS"], string_elem.attrib["VPOS"]
    w, h = string_elem.attrib["WIDTH"], string_elem.attrib["HEIGHT"]
    x, y, w, h = float(x), float(y), float(w), float(h)
    return BBOX(
        x, y, x + w, y + h,
        string_elem.attrib["ID"],
        file=xml_path, image=image_path,
        style=string_elem.get("STYLEREFS")

    )


def get_alto_bboxes(xml, xml_path: str = "", image_path: str = ""):
    bboxes = []
    for string in xml.xpath("//a:String", namespaces=NS):
        bboxes.append(_compute_bbox_form_node(string, xml_path, image_path))
    return bboxes


def read_alto_for_tagging(alto_xml) -> Tuple[List[BBOX], str, et.ElementBase]:
    xml, image_path = _parse_alto(alto_xml)
    bboxes = get_alto_bboxes(xml, alto_xml, image_path)
    return bboxes, image_path, xml


def extract_images_from_bbox_dict_for_training(
    images: Dict[str, Dict[str, List[BBOX]]],
    output_dir: str = "./data/"
):
    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm.tqdm(total=sum([len(x) for class_items in images.values() for x in class_items.values()]))
    for image, bboxes in images.items():
        for cls, items in bboxes.items():
            source = PILImage.open(image)
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
            for id_, bbox in enumerate(items):
                area = source.crop(bbox.coords)
                area.save(
                    os.path.join(
                        output_dir,
                        cls,
                        f"{os.path.basename(bbox.image)}.{id_}.png"
                    )
                )
                pbar.update(1)


def extract_images_from_bbox_dict_for_tagging(
    bboxes:  List[BBOX],
    image: str
):
    """

    Note: Unlink extract_images_from_bbox_dict_for_training, this does one page at a time
    """
    image_list = []
    source = PILImage.open(image)
    for bbox in bboxes:
        area = source.crop(bbox.coords)
        image_list.append(area)
    return image_list


def move_images(source, dest: str, max_size: int, ratio: float = None):
    for cls in glob.glob(source):
        files = glob.glob(os.path.join(cls, "*.png"))
        random.shuffle(files)
        os.makedirs(
            cls.replace(os.path.dirname(source), os.path.dirname(dest)),
            exist_ok=True
        )
        if max_size > 0:
            files = files[:int(max_size * ratio)]
        for file in files:
            os.rename(file, file.replace(os.path.dirname(source), os.path.dirname(dest)))


def split_dataset(source, max_size, dest: str = "./output/", test_val: float = 0.1, dev_val: float = 0.05,
                  except_for_train: bool = False):
    train_dir = os.path.join(dest, "train", "")
    test_dir = os.path.join(dest, "test", "")
    dev_dir = os.path.join(dest, "dev", "")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)

    move_images(source, dest=test_dir, max_size=max_size, ratio=test_val)
    move_images(source, dest=dev_dir, max_size=max_size, ratio=dev_val)
    if except_for_train:
        move_images(source, dest=train_dir, max_size=-1, ratio=None)
