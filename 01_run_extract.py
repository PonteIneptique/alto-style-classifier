import glob
from collections import defaultdict
from stylalto.datasets.extractor import read_alto_for_training, extract_images_from_bbox_dict_for_training, split_dataset


data = defaultdict(list)
images = {}
for xml_path in glob.glob("./input/**/*.xml", recursive=True):
    current, image = read_alto_for_training(xml_path)
    images[image] = current
    for key in current:
        data[key].extend(current[key])

minimum = float("inf")
for cls in data:
    total = sum([len(val) for val in data.values()])
    print(f"{cls.zfill(10).replace('0', ' ')} : {len(data[cls]) / total:.2f} of the whole ({len(data[cls])})")
    minimum = min([len(data[cls]), minimum])

# Extract images
extract_images_from_bbox_dict_for_training(images, output_dir="./data/")

# Split into dataset
split_dataset("./data/*", max_size=minimum, except_for_train=True)
