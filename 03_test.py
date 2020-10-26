from stylalto.trainer import Trainer

trainer = Trainer(
    nb_classes=3,
    preprocess="resize-center",  # resize, center
    model="seqconv",
    batch_size=64,
    device="cuda",
    class_to_idx={
        "_": 0,
        "bold": 1,
        "italics": 2
    }
)

trainer.model.load_from_path("./results/model2.pth")

testset, testloader = trainer.generate_dataset("output/test")
print(f"Test set: {len(testset)}")

test_loss, accuracy, truthes, preds = trainer.eval(
    dataset_loader=testloader,
    return_preds_and_truth=True
)
trainer._print_accuracy("Test set", test_loss, accuracy, testloader.dataset)
trainer.get_eval_details(gts=truthes, preds=preds)
