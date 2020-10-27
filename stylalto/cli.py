import os
from collections import defaultdict

import click
import tqdm

from .datasets.extractor import (
    read_alto_for_training, extract_images_from_bbox_dict_for_training, split_dataset
)
from .tagger import Tagger
from .trainer import Trainer


DEFAULT_TRAINING_DATA_DIR = "./training-data/"


@click.group("stylalto")
def group():
    """ Stylalto, tagger and trainer for recognizing style in ALTO files"""


@group.command("extract")
@click.argument("input", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("--output_dir", default=DEFAULT_TRAINING_DATA_DIR, type=click.Path(file_okay=False, dir_okay=True))
def extract(input_file_paths, output_dir):
    """ Generate dataset for traing, requires valid ALTO files
    """
    data = defaultdict(list)
    images = {}
    for xml_path in input_file_paths:
        current, image = read_alto_for_training(xml_path)
        images[image] = current
        for key in current:
            data[key].extend(current[key])

    minimum = float("inf")
    for cls in data:
        total = sum([len(val) for val in data.values()])
        click.echo(
            f"Class {cls.zfill(10).replace('0', ' ')} : {len(data[cls]) / total:.2f} of the whole ({len(data[cls])})"
        )
        minimum = min([len(data[cls]), minimum])

    # Extract images
    extract_images_from_bbox_dict_for_training(images, output_dir=output_dir)

    # Split into dataset
    split_dataset(os.path.join(output_dir, "*"), max_size=minimum, except_for_train=True)


@group.command("tag")
@click.argument("input", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
def tag(input_files):
    tagger = Tagger.load_from_prefix("example_model/model2")
    for xml in tqdm.tqdm(tagger.tag(input_files, batch_size=4)):
        continue


@group.command("test")
@click.argument("test_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.argument("model_prefix", type=str)#, help="Should be the beginning of a path such as {model}.pth exists")
def test(test_dir, model_prefix):
    trainer = Trainer.load_from_prefix(model_prefix)

    testset, testloader = trainer.generate_dataset(test_dir)
    click.echo(f"Test set: {len(testset)} elements")

    test_loss, accuracy, truthes, preds = trainer.eval(
        dataset_loader=testloader,
        return_preds_and_truth=True
    )
    trainer._print_accuracy("Test set", test_loss, accuracy, testloader.dataset)
    trainer.get_eval_details(gts=truthes, preds=preds)


if __name__ == "__main__":
    group()
