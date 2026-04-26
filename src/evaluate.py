"""
Music Genre Classifier - AI-powered music genre classification
Copyright (C) 2026 Mathilde Pascal

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime

def load_checkpoint(filepath, model):
    """
    Load saved model.

    Args:
        filepath: Path to a saved .pth file
        model   : Uninitialised model instance (e.g. audioCNN())
    
    Returns:
        model   : Model with loaded weights, set to eval mode.
    """
    checkpoint = torch.load(filepath, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model

def model_accuracy(model, classes, test_dataset, test_loader):
    """
    Calculate overall accuracy and class accuracy.

    Args:
        model       : Trained model
        classes     : List of classes
        test_dataset: Test dataset of images and labels
        test_loader : DataLoader object for test dataset

    Returns:
        total_accuracy: overall accuracy (%) of the trained model on the test dataset
        class_accuracy: accuracy (%) of the trained model for each class
        y_true        : Array of actual labels
        y_pred        : Array of predicted labels
    """

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    y_pred = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            __, predictions = torch.max(outputs, 1)
            
            # Totals for calculating accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[int(label)]] += 1
                total_pred[classes[int(label)]] += 1
                y_pred.append(prediction)
                y_true.append(label)

    total_accuracy = correct / total
    class_accuracy = dict()

    print(f'Overall accuracy of the network on {len(test_dataset)} test samples: {100 * total_accuracy} %\n')
    print('Accuracy on each class:\n')

    # pretty print
    max_len = max([len(c) for c in classes])
    spaces = {c: ' '*(max_len - len(c)) for c in classes}

    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        class_accuracy.update({classname: accuracy})
        print(f'{classname}{spaces[classname]}: {100*accuracy:.1f} %')

    return total_accuracy, class_accuracy, y_true, y_pred
    

def plot_test(model, classes, test_dataset, n_show=4):
    """
    Show a row of n_show images from the test dataset with their true label and their predicted label.
    Figure saved to ./outputs/figures/example_{timestamp}.png

    Args:
        model       : Trained model
        classes     : List of classes
        test_dataset: Test dataset of images and labels
        n_show      : Number of examples to show

    Returns:
        None
    """

    dataiter = iter(DataLoader(test_dataset, batch_size=n_show, shuffle=True))
    images, labels = next(dataiter)
    outputs = model(images)
    __, predicted = torch.max(outputs, 1)

    plt.figure()

    for i in range(n_show):
        image = images[i]
        label = labels[i]
        pred_label = predicted[i]
        ax = plt.subplot(1, n_show, i + 1)
        plt.tight_layout()
        ax.set_title('Label: {}\nPred: {}'.format(classes[label], classes[pred_label]))
        ax.axis('off')
        plt.imshow(np.transpose(image, (1, 2, 0)))
    
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join('.', 'outputs','figures', 'examples_'+timestamp+'.png'), dpi=200, format='png')
    plt.show()

    return


def plot_confusion_matrix(classes, y_true, y_pred, annot=True):
    """
    Show the confusion matrix.
    Figure saved to ./outputs/figures/confusion_maxtrix_{timestamp}.png

    Args:
        classes : List of classes
        y_true  : Array of true labels
        y_pred  : Array of predicted labels
        annot   : Annotate the confusion matrix with percentages. Defaults to True.

    Returns:
        None
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                         index = classes,
                         columns = [i for i in classes])
    
    ax = sns.heatmap(df_cm, annot=annot, fmt='.0%', cmap='Blues')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_title("Confusion matrix")
    plt.tight_layout()

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join('.', 'outputs','figures', 'confusion_matrix_'+timestamp+'.png'), dpi=200, format='png')

    plt.show()
    return