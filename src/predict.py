import torch
import os
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as display
import torchvision.transforms as transforms
import librosa

if __name__ == "__main__":
    from model import audioCNN, audioCNN2
    from evaluate import load_checkpoint
else:
    from .model import audioCNN, audioCNN2
    from .evaluate import load_checkpoint

classes = os.listdir(os.path.join("data","genres_original"))

# We are using the best-performing CNN architecture that we tested, which contains 3 convolution layers.
model_classes = {'audioCNN': audioCNN, 'audioCNN2': audioCNN2}

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    architecture = checkpoint['architecture']
    model = load_checkpoint(checkpoint_path, model_classes[architecture]())

    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model

def predict_genre(model, filepath, name):
    filename = os.path.join(filepath, name)

    # Create temporary directory to save spectrograms to
    if not os.path.exists(os.path.join("data","tmp_dir")):
        os.mkdir(os.path.join("data","tmp_dir"))
    tmp_dir = os.path.join("data", "tmp_dir")

    # Load audio file and convert to spectrogram
    y, sr = librosa.load(filename)

    # Crop file length to an integer number of seconds
    y_end = int(np.round((len(y)/sr - np.floor(len(y)/sr)) * sr))
    y = y[:-y_end]

    duration = librosa.get_duration(y=y, sr=sr) # total audio duration
    seg_duration = 30 # duration of each segment - must match what the model was trained on!
    seg_length = int(sr * seg_duration) # length of each segment = sampling rate * seconds
    n_segments = int(np.floor(duration/seg_duration) + 1) # total number of segments with small overlap between segments
    if n_segments > 1:
        overlap = int(sr * (n_segments * seg_duration - duration) / (n_segments - 1)) # length of overlap
    else:
        overlap = 0

    for i in range(n_segments):
        # Create 30-second segment
        t = y[i * (seg_length - overlap): i * (seg_length - overlap) + seg_length]

        # Convert to spectrogram
        spec = librosa.feature.melspectrogram(y=t, sr=sr)
        spec_dB = librosa.amplitude_to_db(spec, ref=np.max)
        
        # Save to png file
        dpi = 72
        width_px = 432
        height_px = 288
        figsize = (width_px/dpi, height_px/dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        img = librosa.display.specshow(spec_dB, sr=sr, fmax=sr/2, ax=ax)
        ax.axis('off')
        plt.savefig(os.path.join(tmp_dir, f"seg{i+1}.png"),
                    dpi=dpi,
                    format='png')
        plt.close(fig)

    # List of all model outputs: for each segment, a list of the probability of belonging to each class
    all_probs = []

    for img_name in os.listdir(tmp_dir):
        img = Image.open(os.path.join(tmp_dir, img_name)).convert('RGB') 
        img = transforms.ToTensor()(img)
        output = model(img)

        # Convert output to a probability
        output_proba = torch.nn.Softmax(dim=1)(output)
        all_probs.append(output_proba)
    
    all_probs = torch.cat(all_probs, dim=0)
    avg_probs = all_probs.mean(dim=0) # Average probability of belonging to each class

    final_prediction = avg_probs.argmax().item() # Class with the highest probability
    confidence = avg_probs.max().item() # Probability of the most likely class

    print(f"Predicted genre:  {classes[final_prediction]}")
    print(f"with probability: {confidence*100:.2f}%")
    print("="*30)
    print("All predictions:")
    print("="*30)

    max_len = max([len(c) for c in classes])
    spaces = {c: ' '*(max_len - len(c)) for c in classes}

    sorted_probas = reversed(np.argsort(avg_probs))

    for i in sorted_probas:
        print(f"{classes[i]}:{spaces[classes[i]]}\t{avg_probs[i]*100:.2f}%")

if __name__ == "__main__":
    # Loading best performing model
    # May require changing file name
    checkpoint_path = os.path.join("outputs", "models", "checkpoint_audioCNN_best.pth")
    model = load_model(checkpoint_path)

    # Example using a random example drawn from a directory containing audio files

    # Directory of audio tracks to test
    sample_dir = os.path.join("data", "test_samples")

    i = random.randint(0,len(os.listdir(sample_dir))-1)
    filename = os.listdir(sample_dir)[i]
    print(filename)
    predict_genre(model, sample_dir, filename)
    display.display(display.Audio(os.path.join(sample_dir,filename)))