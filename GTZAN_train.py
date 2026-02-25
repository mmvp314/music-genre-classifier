import torch.nn as nn
import torch.optim as optim
import argparse

from src.dataset import classes, get_datasets, get_dataloaders
from src.model import audioCNN
from src.train import train_CNN, plot_history

def parse_args():
    parser = argparse.ArgumentParser(description="Train a music genre classifier.")
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=20)
    return parser.parse_args()

if __name__ == "__main__":
    print(f"Genres in the GTZAN dataset: {classes}")

    args = parse_args()

    model = audioCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_dataset, val_dataset, test_dataset = get_datasets(
                                    data_dir=args.data_dir,
                                    val_split=args.val_split,
                                    test_split=args.test_split
                                    )
    
    train_loader, val_loader, __ = get_dataloaders(train_dataset,
                                    val_dataset,
                                    test_dataset,
                                    batch_size=args.batch_size
                                    )
    
    history = train_CNN(model, train_loader, val_loader, criterion, optimizer, epochs=args.epochs)
    plot_history(history)
