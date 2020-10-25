from typing import Dict, List, Tuple
from collections import namedtuple, defaultdict
import lxml.etree as et
import os
import PIL.Image as PILImage
import tqdm
import random
import glob


class NoSourceImage(Exception):
    """ Raised when the ALTO is missing a link to an image file """


BBOX = namedtuple("Bbox", ["x1", "y1", "x2", "y2", "file", "id", "image"])
NS = {"a": "http://www.loc.gov/standards/alto/ns-v2#"}


def temporary_replace_path(xml_path):
    return f"../IMG/{xml_path.replace('.xml', '.jpg')}"


def read_alto(alto_xml) -> Tuple[Dict[str, List[BBOX]], str]:
    classes = defaultdict(list)

    with open(alto_xml) as f:
        xml = et.parse(f)
        source_image = xml.xpath("//a:sourceImageInformation/a:fileName/text()", namespaces=NS)
        if not len(source_image):
            raise NoSourceImage(f"{alto_xml} is missing the following node"
                                "`/alto/Description/sourceImageInformation/fileName`"
                                "which should contain the path to the image it is about")
        source_image = temporary_replace_path(source_image[0])
        source_image_real_path = os.path.abspath(
            os.path.join(os.path.dirname(alto_xml), source_image)
        )
        if not os.path.isfile(source_image_real_path):
            raise NoSourceImage(f"{alto_xml} has a wrong path at"
                                "`/alto/Description/sourceImageInformation/fileName`"
                                f": {source_image_real_path}")

        styles = {
            style.attrib["ID"]: style.attrib["FONTSTYLE"] if style.attrib["FONTSTYLE"] else "_"
            for style in xml.xpath("//a:TextStyle", namespaces=NS)
        }
        for string in xml.xpath("//a:String", namespaces=NS):
            x, y, = string.attrib["HPOS"], string.attrib["VPOS"]
            w, h = string.attrib["WIDTH"], string.attrib["HEIGHT"]
            x, y, w, h = float(x), float(y), float(w), float(h)
            style = styles[string.attrib["STYLEREFS"]]
            classes[style].append(BBOX(x, y, x + w, y + h, alto_xml,
                                       string.attrib["ID"], source_image_real_path))

    return classes, source_image_real_path


def extract_images_from_bbox_dict(
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
                area = source.crop(bbox[:4])
                area.save(
                    os.path.join(
                        output_dir,
                        cls,
                        f"{os.path.basename(bbox.image)}.{id_}.png"
                    )
                )
                pbar.update(1)


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
