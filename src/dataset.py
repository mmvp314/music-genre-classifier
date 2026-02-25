import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import librosa
import random

audio_dir = './data/genres_original' # directory of audio (.wav) files
im_dir = './data/images_original' # directory of spectrogram (.png) files

classes = os.listdir(audio_dir)

# Note that file jazz.00054.wav from the Kaggle download is corrupted so we delete it.
if os.path.exists(os.path.join(audio_dir, 'jazz','jazz.00054.wav')):
    os.remove(os.path.join(audio_dir, 'jazz','jazz.00054.wav'))

'''
# The GTZAN dataset already includes spectrograms.
# Here is the code to recreate them from the audio files
# and store them in a folder 'images' in the 'data' folder.

os.makedirs('./data/images', exist_ok=True)
for subd in os.listdir(audio_dir):
    os.makedirs(f'./data/images/{subd}', exist_ok=True)

data = [(file_name, subd) 
        for subd in os.listdir(audio_dir) 
        for file_name in os.listdir(os.path.join(audio_dir, subd))]

for i in range(len(data)):
    file_name, label = data[i]
    y, sr = librosa.load(os.path.join(audio_dir, label, file_name))
    spec = librosa.feature.melspectrogram(y=y, sr=sr)

    fig, ax = plt.subplots()
    spec_dB = librosa.amplitude_to_db(spec, ref=np.max)
    img = librosa.display.specshow(spec_dB, sr=sr, fmax=sr/2, ax=ax)

    plt.savefig(f'./data/images/{label}/{file_name[:-4]}.png', dpi=72, format='png',bbox_inches='tight')
'''

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor()
    ])

def get_datasets(data_dir="./data",
                 val_split=0.1,
                 test_split=0.1,
                 seed=10):
    """
    Split and return train/val/test Datasets.

    Args:
        data_dir:    Where to store/load the dataset. Defaults to ./data
        val_split:   Fraction of training data to use for validation. Defaults to 0.1.
        test_split:  Fraction of training data to use for testing. Defaults to 0.1.
        seed:        Random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    
    Note:
        The dataset must already been downloaded into the 'data' folder.
        The 'data' folder must contain 2 subfolders: 'genres_original' and 'images_original'.
        The folders containing the training, test, and validation datasets are overwritten 
        when this script is called.
    """

    transform = get_transforms()
    random.seed(seed)

    # Split data into 'train', 'test' and 'val' folders, each containing one folder per genre.
    # This allows us to use the ImageFolder constructor to create each subdataset.
    folder_names = [os.path.join(data_dir, 'train'),
                    os.path.join(data_dir, 'val'),
                    os.path.join(data_dir, 'test')]

    train_dir, val_dir, test_dir = folder_names

    # Create train/val/test subfolders in ./data folder
    for f in folder_names:
        if os.path.exists(f):
            shutil.rmtree(f)
        os.mkdir(f)

    for i, g in enumerate(classes):

        # Shuffle full dataset
        file_names = os.listdir(os.path.join(im_dir, g))
        random.Random(i).shuffle(file_names)

        # Split into test/val/train subdatasets
        test_files = file_names[:int(test_split * len(file_names))]
        val_files = file_names[int(test_split * len(file_names)):int((test_split + val_split) * len(file_names))]
        train_files = file_names[int((test_split + val_split) * len(file_names)):]

        # Create a subfolder for each genre in train/val/test folders
        for f in folder_names:
            os.mkdir(os.path.join(f, g))

        # Copy images into folders
        for f in train_files:
            shutil.copy(os.path.join(im_dir,g,f), os.path.join(data_dir,'train',g))
        for f in val_files:
            shutil.copy(os.path.join(im_dir,g,f), os.path.join(data_dir,'val',g))
        for f in test_files:
            shutil.copy(os.path.join(im_dir,g,f), os.path.join(data_dir,'test',g))

    # Create dataset objects; labels are inferred from the directory structure
    train_dataset = datasets.ImageFolder(train_dir, transform)
    val_dataset = datasets.ImageFolder(val_dir, transform)
    test_dataset = datasets.ImageFolder(test_dir, transform)

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset,
                    val_dataset,
                    test_dataset,
                    batch_size=16,
                    seed=10):
    """
    Return train/val/test DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset  : Validation dataset
        test_dataset : Test dataset
        batch_size   : Size of batches into which the dataset is split. Defaults to 16.
        seed         : Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# Sanity check — run this file directly to verify everything loads correctly
if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    print(f"Training items : {len(train_dataset)}")
    print(f"Val items      : {len(val_dataset)}")
    print(f"Test items     : {len(test_dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape    : {images.shape}")   # Expected: [batch_size, 3, 288, 432]
