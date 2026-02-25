import os
import torch
import argparse

from src.dataset import classes, get_datasets, get_dataloaders
from src.model import audioCNN, audioCNN2
from src.evaluate import load_checkpoint, model_accuracy, plot_test, plot_confusion_matrix

model_classes = {'audioCNN': audioCNN, 'audioCNN2': audioCNN2}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--checkpoint", type=str,   required=True, 
                        help="Filename of the checkpoint to load from the outputs/models folder, e.g. checkpoint_audioCNN_20240101_120000.pth")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_dataset, val_dataset, test_dataset = get_datasets(
                                        data_dir=args.data_dir,
                                        val_split=args.val_split,
                                        test_split=args.test_split
                                        )
    
    __, __, test_loader = get_dataloaders(train_dataset,
                                        val_dataset,
                                        test_dataset,
                                        batch_size=args.batch_size
                                        )

    checkpoint_path = os.path.join(".", "outputs", "models", args.checkpoint)
    architecture = torch.load(checkpoint_path, weights_only=True)['architecture']
    model = load_checkpoint(checkpoint_path, model_classes[architecture]())

    # Print model accuracy and load arrays of true and predicted labels    
    __, __, y_true, y_pred = model_accuracy(model, classes, test_dataset, test_loader)

    # Show some examples on the test dataset
    plot_test(model, classes, test_dataset)

    # Plot the confusion matrix
    plot_confusion_matrix(classes, y_true, y_pred)