# Music genre classification

This project uses a Convolutional Neural Network to classify music genres, using the GTZAN dataset.[^1] The data consists of 30-second audio (.wav) files representing 10 music genres: blues, classical, country, disco, hip hop, jazz, metal, pop, reggae, and rock.

## Setup

### 1. Pull the code from GitHub

In a command prompt window, navigate to the folder where you want to store the project (replace `path` with the actual path, e.g. `C:\Users\your-username\Documents`):
```
cd path
```

Clone the repository and navigate into it:
```
git clone https://github.com/mmvp314/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Create the data and outputs folders

The repository does not include the `data` and `outputs` folders. Create them manually:
```
mkdir data
mkdir outputs
mkdir outputs\figures
mkdir outputs\models
```

Your directory tree should now look like this:
```
music-classification
├───data
├───outputs
│   ├───figures
│   └───models
└───src
```

### 3. Download the dataset

Download and extract the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) into the `data` folder. The structure should now look like this:
```
music-classification
├───data
│   ├───genres_original
│   │   ├───blues
│   │   ├───classical
│   │   ├───country
│   │   ├───disco
│   │   ├───hiphop
│   │   ├───jazz
│   │   ├───metal
│   │   ├───pop
│   │   ├───reggae
│   │   └───rock
│   └───images_original
│       ├───blues
│       ├───classical
│       ├───country
│       ├───disco
│       ├───hiphop
│       ├───jazz
│       ├───metal
│       ├───pop
│       ├───reggae
│       └───rock
├───outputs
│   ├───figures
│   └───models
└───src
```

### 4. Create a Python environment

Create a virtual environment called `venv` (or any name you prefer):
```
python -m venv venv
```

Activate it:
```
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your command prompt line.

### 5. Install dependencies

```
pip install -r requirements.txt
```

## How to run

To launch a training run, execute this line with any arguments detailed in `GTZAN_train.py`, for instance:
```
python GTZAN_train.py --architecture audioCNN --epochs 20
```

To evaluate a saved checkpoint, pass its filename via the `--checkpoint` argument:
```
python GTZAN_evaluate.py --checkpoint checkpoint_audioCNN_20240101_120000.pth
```

Saved checkpoints are in `outputs\models`. Figures are saved to `outputs\figures`.

## Results

| Model     | Architecture  | Batch size | Max epochs | Optimal epoch | Test accuracy |
|-----------|---------------|------------|------------|---------------|---------------|
| audioCNN  | 3 conv layers | 32         | 50         | 12            | 58.59%        |
| audioCNN  | 3 conv layers | 64         | 20         | 20            | 59.60%        |
| audioCNN2 | 2 conv layers | 64         | 20         | 19            | 42.42%        |
| audioCNN  | 3 conv layers | 128        | 50         | 30            | 62.63%        |


The optimal epoch is the epoch at which the highest validation accuracy is achieved and is indicated by a red dot on the training history plot.

## Bonus

The "bonus" notebooks provide an overview of how basic implementations of other ML classification models perform:
- `bonus_svm.ipynb`: Principal Component Analysis (PCA) and Standard Vector Machine (SVM)
- `bonus_random_forest.ipynb`: Random forest (work in progress)
- `bonus_xgboost.ipynb`: XGBoost (work in progress)

[^1]: George Tzanetakis, Georg Essl, and Perry Cook. Automatic musical genre classification of audio signals. 2001. URL: http://ismir2001.ismir.net/pdf/tzanetakis.pdf.