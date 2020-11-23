import os
from collections import defaultdict
import glob
import sys
from typing import List

import click
import tqdm
import torch.cuda
import lxml.etree as et

from .datasets.extractor import (
    read_alto_for_training, extract_images_from_bbox_dict_for_training, split_dataset
)
from .errors import StylaltoException
from .preprocesses import PREPROCESSES
from .tagger import Tagger
from .trainer import Trainer, MODELS
from .viz import vizualise_from_file


DEFAULT_TRAINING_DATA_DIR = "./training-data/"


def expand_paths(paths: List[str], recursive: bool = False):
    read = []
    for file in paths:
        if os.path.isfile(file):
            read.append(file)
        else:
            read.extend(
                [
                    f
                    for f in glob.glob(file, recursive=recursive)
                    if os.path.isfile(f)
                ]
            )
    return read


@click.group("stylalto")
def group():
    """ Stylalto, tagger and trainer for recognizing style in ALTO files"""


@group.command("extract")
@click.argument("input_file_paths", nargs=-1)
@click.option("--output_dir", default=DEFAULT_TRAINING_DATA_DIR, type=click.Path(file_okay=False, dir_okay=True))
@click.option("--dry", default=False, is_flag=True, help="Does not run extract and shows all problem")
@click.option("--recursive", default=False, is_flag=True, help="Search for files recursively")
@click.option("--use_minimum", default=False, is_flag=True, help="Use a non bias corpus by splitting around the "
                                                                 "smallest corpus size")
@click.option("--test", default=False, is_flag=True, help="Do not extract, just test extraction")
def extract(input_file_paths, output_dir, dry=False, recursive=False, use_minimum=False, test=True):
    """ Generate dataset for training, requires valid ALTO files. INPUT_FILE_PATHS can be *quoted* UNIX globs.

    """
    if test:
        click.secho("Entering test mode")
        errors = 0
        uncaught = 0
        dry = True

    data = defaultdict(list)
    images = {}

    read = expand_paths(input_file_paths, recursive=recursive)

    for xml_path in tqdm.tqdm(sorted(list(set(read)))):
        try:
            current, image = read_alto_for_training(xml_path)
        except StylaltoException as E:
            click.secho(str(E), fg="red")
            if not dry:
                return None
            if test:
                errors += 1
        except Exception as E:
            click.secho(f"{xml_path} failed", fg="red")
            if test:
                uncaught += 1
                continue
            raise
        images[image] = current
        for key in current:
            data[key].extend(current[key])

    if dry:
        if test:
            failed = errors+uncaught >= 1
            click.echo("\n\n------------------\n\nEnd of tests")
            if failed:
                click.secho(f"Failures: {errors}, Unknown errors for Stylalto: {uncaught}", fg="red")
            sys.exit(int(failed))
        click.secho("End of dry run", fg="green")
        return

    minimum = float("inf") if use_minimum else None
    for cls in data:
        total = sum([len(val) for val in data.values()])
        click.echo(
            f"Class {cls.zfill(10).replace('0', ' ')} : {len(data[cls]) / total:.2f} of the whole ({len(data[cls])})"
        )
        if use_minimum:
            minimum = min([len(data[cls]), minimum])

    # Extract images
    extract_images_from_bbox_dict_for_training(images, output_dir=output_dir)

    # Split into dataset
    split_dataset(os.path.join(output_dir, "*"), max_size=minimum, except_for_train=True)


@group.command("tag")
@click.argument("model_prefix", type=str)
@click.argument("input_files", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("--viz", type=bool, is_flag=True)
def tag(model_prefix, input_files, viz: bool = False, no_tag: bool = False):
    tagger = Tagger.load_from_prefix(model_prefix)
    for file, xml in tqdm.tqdm(tagger.tag(input_files, batch_size=4)):
        if viz:
            figure = vizualise_from_file(xml, xml_filepath=file)
            figure.savefig(f"{file}.result-viz.png")
            click.echo(f"Saved viz in {file}.result-viz.png")
        continue


@group.command("viz")
@click.argument("input_files", nargs=-1)
@click.option("--recursive", default=False, is_flag=True, help="Search for files recursively")
def vizualize(input_files, recursive):
    input_files = expand_paths(input_files, recursive=recursive)

    for file in tqdm.tqdm(input_files):
        with open(file) as f:
            xml = et.parse(file)
        figure = vizualise_from_file(xml, xml_filepath=file)
        figure.savefig(f"{file}.result-viz.png")
        click.echo(f"Saved viz in {file}.result-viz.png")


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


@group.command("train")
@click.argument("train_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument("dev_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option("--test_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True), default=None,
              help="Test directory to use")
@click.option("--model", default="seqconv", type=click.Choice(list(MODELS.keys())),
              help="Model architecture to use")
@click.option("--preprocess", default="resize-center", type=click.Choice(list(PREPROCESSES.keys())),
              help="Preprocess function")
@click.option("--batch", type=int, default=16,
              help="Batch size")
@click.option("--epochs", type=int, default=50,
              help="Number of epochs")
@click.option("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
              help="Device to train on")
@click.option("--optimizer", type=click.Choice(["Adam", "SGD"]), default="Adam",
              help="Optimizer to use")
@click.option("--path_prefix", type=str, default="./results/model",
              help="Path and prefix such as prefix=./results/model will create "
                   "./results/model.pth and ./results/model_config.json")
def train(train_dir, dev_dir, test_dir, model, preprocess, batch, epochs, device, optimizer, path_prefix):

    nb_classes = 0

    for path in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, path)):
            nb_classes += 1

    trainer = Trainer(
        nb_classes=nb_classes,
        preprocess=preprocess,  # resize, center
        model=model,
        batch_size=batch,
        device=device
    )

    trainer.train(
        n_epochs=epochs,
        optimizer=optimizer,
        dev_dir=dev_dir,
        train_dir=train_dir,
        prefix_model_name=path_prefix
    )

    if test_dir:
        testset, testloader = trainer.generate_dataset(test_dir)
        print(f"Test set: {len(testset)}")

        test_loss, accuracy, truthes, preds = trainer.eval(
            dataset_loader=testloader,
            return_preds_and_truth=True
        )
        trainer._print_accuracy("Test set", test_loss, accuracy, testloader.dataset)
        trainer.get_eval_details(gts=truthes, preds=preds)


if __name__ == "__main__":
    group()
