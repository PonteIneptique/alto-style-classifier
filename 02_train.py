from stylalto.trainer import Trainer
import torch.cuda

trainer = Trainer(
    nb_classes=3,
    preprocess="resize-center",  # resize, center
    model="seqconv",
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

model = trainer.train(
    n_epochs=200,
    optimizer="Adam",
    dev_dir="output/dev",
    train_dir="output/train",
)

testset, testloader = trainer.generate_dataset("output/test")
print(f"Test set: {len(testset)}")

test_loss, accuracy, truthes, preds = trainer.eval(
    dataset_loader=testloader,
    return_preds_and_truth=True
)
trainer._print_accuracy("Test set", test_loss, accuracy, testloader.dataset)
trainer.get_eval_details(gts=truthes, preds=preds)
