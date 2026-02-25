# Music genre classification

This project uses a Convolutional Neural Network to classify music genres, using the GTZAN dataset.[^1] The data consists of 30-second audio (.wav) files representing 10 music genres: blues, classical, country, disco, hip hop, jazz, metal, pop, reggae, and rock.

## Setup
Here is what the initial directory tree should look like:
```
music-classification
├───data
├───outputs
│   ├───figures
│   └───models
└───src
```
Once you have created this structure, download and extract the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) into the data folder. The structure should now look like this:
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

## How to run this code

The prompts below work for the Windows Command Prompt. Refer to documentation for other operating systems.

In a command prompt window, navigate to your subfolder `music-classification` (replace `path` with the actual path e.g. `C:\Users\your-username\Documents\music-classification`):
```
$ cd path\music-classification
```

To launch a training run, execute this line with any arguments detailed in the `GTZAN_train.py` file, for instance:
```
$ python GTZAN_train.py --epochs 20
```

To launch the evaluation, execute this line:
```
$ python GTZAN_evaluate.py
```

When prompted, select the model you want to evaluate. A time stamp is appended to the model name so you can easily find the latest model run.

Models are saved to `./outputs/models`. Figures are saved to `./outputs/figures`.


[^1]: George Tzanetakis, Georg Essl, and Perry Cook. Automatic musical genre classification of audio signals. 2001. URL: http://ismir2001.ismir.net/pdf/tzanetakis.pdf.