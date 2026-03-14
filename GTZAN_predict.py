import os
import torch
import argparse
import IPython.display as display

from src.predict import load_model, predict_genre

def parse_args():
    parser = argparse.ArgumentParser(description="Predict genre of any music file")
    parser.add_argument("--filepath",        type=str, 
                        help="Path to the folder in which the audio file is located")
    parser.add_argument("--name",            type=str, 
                        help="Audio file name (including extension e.g. 'track.mp3')")
    parser.add_argument("--checkpoint-path", type=str, 
                        default=os.path.join("outputs","models"),
                        help="Path to folder containing model checkpoint. Default: ./outputs/models")
    parser.add_argument("--checkpoint-name", type=str, 
                        help="File name of model checkpoint (if none available, run GTZAN_train.py)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    c_path = os.path.join(args.checkpoint_path, args.checkpoint_name)
    model = load_model(c_path)
    filename = os.path.join(args.filepath, args.name)
    predict_genre(model, args.filepath, args.name)
    display.display(display.Audio(filename))