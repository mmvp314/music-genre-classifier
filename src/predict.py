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

import matplotlib
matplotlib.use('Agg')
import torch
import os
import sys
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import librosa
import librosa.display

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import audioCNN, audioCNN2
    from evaluate import load_checkpoint
else:
    from .model import audioCNN, audioCNN2
    from .evaluate import load_checkpoint

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 
           'jazz', 'metal', 'pop', 'reggae', 'rock']

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

def create_spectrogram_from_audio(audio_segment, sr):
    """
    Create spectrogram image from audio segment (in-memory)
    
    Args:
        audio_segment: numpy array of audio samples
        sr: sample rate
    
    Returns:
        PIL Image object (RGB)
    """
    # Convert to mel spectrogram
    spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr)
    spec_dB = librosa.amplitude_to_db(spec, ref=np.max)
    
    # Create figure with dimensions matching training data (432x288 pixels)
    dpi = 72
    width_px = 432
    height_px = 288
    figsize = (width_px/dpi, height_px/dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display spectrogram
    librosa.display.specshow(spec_dB, sr=sr, fmax=sr/2, ax=ax)
    ax.axis('off')
    
    # Save to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, format='png')
    plt.close(fig)
    
    # Convert buffer to PIL Image
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    
    return img

def predict_genre_from_bytes(model, audio_bytes, filename):
    """
    Predict genre from audio file bytes (in-memory processing)
    
    Args:
        model: trained PyTorch model
        audio_bytes: bytes object of audio file
        filename: original filename (for reference)
    
    Returns:
        dict with prediction results
    """

    try:
        # Load audio from bytes
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)

        try:
            y, sr = librosa.load(audio_buffer, sr=None)
        except Exception as load_error:
            # If BytesIO fails, try with file extension hint
            audio_buffer = io.BytesIO(audio_bytes)
            file_ext = filename.split('.')[-1].lower()
            
            # Save temporarily to read (fallback for problematic formats)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            try:
                y, sr = librosa.load(tmp_path, sr=None)
            finally:
                os.unlink(tmp_path)  # Clean up temp file
        
        # Crop file length to an integer number of seconds
        y_end = int(np.round((len(y)/sr - np.floor(len(y)/sr)) * sr))
        if y_end > 0:
            y = y[:-y_end]

        duration = librosa.get_duration(y=y, sr=sr) # total audio duration
        seg_duration = 30 # duration of each segment - must match what the model was trained on!
        seg_length = int(sr * seg_duration) # length of each segment = sampling rate * seconds
        n_segments = int(np.ceil(duration/seg_duration)) # total number of segments

        segment_tensors = []

        for i in range(n_segments):
            # Create 30-second segment
            segment = y[i * seg_length:(i+1) * seg_length]

            if len(segment) < seg_length:
                segment = np.pad(segment, (0, seg_length - len(segment)), mode='constant')

            img = create_spectrogram_from_audio(segment, sr)
            img_tensor = transforms.ToTensor()(img)
            segment_tensors.append(img_tensor)

        batch = torch.stack(segment_tensors)

        with torch.no_grad():
            outputs = model(batch)
            all_probs = torch.nn.functional.softmax(outputs, dim=1)
        
        avg_probs = all_probs.mean(dim=0) # Average probability of belonging to each class

        final_prediction_idx = avg_probs.argmax().item() # Index of class with the highest probability
        confidence = avg_probs.max().item() # Probability of the most likely class

        # Create probability dictionary
        all_probabilities = {classes[i]: float(avg_probs[i]) for i in range(len(classes))}

        sorted_probs = sorted(all_probabilities.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
        result = {
                'predicted_genre': classes[final_prediction_idx],
                'confidence': float(confidence),
                'all_probabilities': dict(sorted_probs),
                'num_segments': n_segments,
                'duration': float(duration)
            }
        
        return result

    except Exception as e:
        import soundfile 
        
        if isinstance(e, soundfile.LibsndfileError) or "Format not recognised" in str(e):
            raise Exception(f"Invalid or corrupted audio file. The file '{filename}' is not a valid audio format or is corrupted. Please upload a valid MP3, WAV, FLAC, or OGG file.")
        elif "shape" in str(e) and "invalid for input" in str(e):
            raise Exception(f"Model input size mismatch. This indicates the spectrogram was generated at an incorrect size.")
        else:
            raise Exception(f"Prediction error: {str(e)}")
    

def predict_genre(model, filepath, name):
    """
    Original function for command-line use (file-based)
    Kept for backwards compatibility with GTZAN_predict.py
    """
    filename = os.path.join(filepath, name)
    
    # Read file into bytes
    with open(filename, 'rb') as f:
        audio_bytes = f.read()
    
    # Use in-memory function
    result = predict_genre_from_bytes(model, audio_bytes, name)
    
    # Print results
    print(f"Predicted genre:  {result['predicted_genre']}")
    print(f"with probability: {result['confidence']*100:.2f}%")
    print(f"Duration: {result['duration']:.1f}s ({result['num_segments']} segments)")
    print("="*30)
    print("All predictions:")
    print("="*30)
    
    for genre, prob in result['all_probabilities'].items():
        print(f"{genre}:\t{prob*100:.2f}%")
    
    return result

if __name__ == "__main__":
    import random
    import IPython.display as display

    # Loading best performing model
    checkpoint_path = os.path.join("outputs", "models", "checkpoint_audioCNN_best.pth")
    model = load_model(checkpoint_path)

    # Example using a random example drawn from a directory containing audio files
    sample_dir = os.path.join("data", "test_samples")

    if os.path.exists(sample_dir) and os.listdir(sample_dir):
        i = random.randint(0,len(os.listdir(sample_dir))-1)
        filename = os.listdir(sample_dir)[i]
        print(f"Testing: {filename}")
        predict_genre(model, sample_dir, filename)
        display.display(display.Audio(os.path.join(sample_dir,filename)))
    else:
        print(f"No test samples found in {sample_dir}")