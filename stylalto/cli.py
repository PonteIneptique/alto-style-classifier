import os
from collections import defaultdict

import click
import tqdm
import torch.cuda

from .datasets.extractor import (
    read_alto_for_training, extract_images_from_bbox_dict_for_training, split_dataset
)
from .errors import StylaltoException
from .preprocesses import PREPROCESSES
from .tagger import Tagger
from .trainer import Trainer, MODELS
from .viz import vizualise_from_file


DEFAULT_TRAINING_DATA_DIR = "./training-data/"


@click.group("stylalto")
def group():
    """ Stylalto, tagger and trainer for recognizing style in ALTO files"""


@group.command("extract")
@click.argument("input_file_paths", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("--output_dir", default=DEFAULT_TRAINING_DATA_DIR, type=click.Path(file_okay=False, dir_okay=True))
@click.option("--dry", default=False, is_flag=True, help="Does not run extract and shows all problem")
@click.option("--use_minimum", default=False, is_flag=True, help="Does not run extract and shows all problem")
def extract(input_file_paths, output_dir, dry=True, use_minimum=True):
    """ Generate dataset for traing, requires valid ALTO files
    """
    data = defaultdict(list)
    images = {}
    for xml_path in input_file_paths:
        try:
            current, image = read_alto_for_training(xml_path)
        except StylaltoException as E:
            print(E)
            if not dry:
                return None
        except Exception as E:
            print(f"{xml_path} failed")
            raise
        images[image] = current
        for key in current:
            data[key].extend(current[key])

    if dry:
        click.echo("End of dry run")
        return

    minimum = float("inf")
    for cls in data:
        total = sum([len(val) for val in data.values()])
        click.echo(
            f"Class {cls.zfill(10).replace('0', ' ')} : {len(data[cls]) / total:.2f} of the whole ({len(data[cls])})"
        )
        if use_minimum:
            minimum = min([len(data[cls]), minimum])
        else:
            minimum = max([len(data[cls]), minimum])

    # Extract images
    extract_images_from_bbox_dict_for_training(images, output_dir=output_dir)

    # Split into dataset
    split_dataset(os.path.join(output_dir, "*"), max_size=minimum, except_for_train=True)


@group.command("tag")
@click.argument("model_prefix", type=str)
@click.argument("input_files", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("--viz", type=bool, is_flag=True)
def tag(model_prefix, input_files, viz: bool = False):
    tagger = Tagger.load_from_prefix(model_prefix)
    for file, xml in tqdm.tqdm(tagger.tag(input_files, batch_size=4)):
        if viz:
            figure = vizualise_from_file(xml, xml_filepath=file)
            figure.savefig(f"{file}.result-viz.png")
            click.echo(f"Saved viz in {file}.result-viz.png")
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
