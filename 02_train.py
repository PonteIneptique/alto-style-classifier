from stylalto.trainer import Trainer


trainer = Trainer(
    dev_dir="output/dev",
    test_dir="output/test",
    train_dir="output/train",
    preprocess="center"  # resize, center
)

trainer.train(
    n_epochs=200
)
