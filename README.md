# ML-Project2

This project applies the Recurrent Neural Network architecture to the music field with the aim of using the [Annotated Beethoven Corpus (ABC)](https://github.com/DCMLab/ABC), developed by the Digital and Cognitive Musicology Laboratory of EPFL, to learn the underlying structure of the chord sequences in Beethoven's string quartets. Our model is able to predict the chords that follow a given sequence of arbitrary length provided by the user and it also offers the option to condition the predicted chords by some features like the global key or the length of the phrases.

This is the second graded project for Machine Learning course (CS-443) at EPFL.

* [Getting started](#getting-started)
    * [Data](#data)
    * [Dependencies](#dependencies)
    * [Report](#report)
* [Running the code](#running-the-code)
    * [Train the model](#train-the-model)
    * [Tensorboard](#tensorboard)
    * [Predict](#predict)

## Getting started
#### Data
The raw data can be downloaded form the publicly available GitHub repository of the [Annotated Beethoven Corpus (ABC)](https://github.com/DCMLab/ABC/data). Following the previous link, you can find two folders `mscx/` and `tsv/` which probidethe annotations of the experts in the digital musical score in MuseScore format, and the extracted annotations as dataframes to ease the usage of the dataset. For this project, we only need the file with all annotations that can be downloaded [here](https://github.com/DCMLab/ABC/blob/master/data/all_annotations.tsv). The default directory to locate the .tsv file is `data/`, which is an empty folder by now. Nevertheless, you can change this location by modifying the default parameter `'data path'` located in the file `utils/params.py` and use your own path.

The processed data and some auxiliary files needed for the cleaning process (such as a JSON file for the mapping from strings to floating point values), will be stored in this same directory.

#### Dependencies
Before trying to run the code, make sure to install all the requirements specified in the `requirements.txt` file. If you use the pip management system, you can just place in the root directory of this project after cloning it to your machine and type:

    pip install -r requirements.txt

#### Report
Check our paper located at the `report/` folder to have an overview of the project and to take a look at the justification of all the steps that led us to the final model.

## Running the code
#### Train the model
Once you have set your environment and all the requirements are installed, you can run the code with the following command to generate our best model. We recommend to include the `--verbose` option to get additional information of the running.

    python run.py --verbose --split_by_phrase

You can also modify other parameters specified with default values at `utils/params.py` or see all the options that can be chosen from the command line by executing one of the two following instructions:

    python run.py -h
    python run.py --help

#### Tensorboard
The log of all prints as well as the parameters of the best model and a `tensorboard` folder, will be located in a folder named `results/experiment_name/`, where the `experiment_name` keyword will be substituted by a string that indicates the parameters of the experiment that affect data processing. In order to see the evolution of the loss curves and the embedding projections, you can run Tensorboard while your code is running to see live evolution or when it has already finished. In order to do so, just go the folder of one experiment and run the following command.

    tensorboard --logdir='tboard'

You can see an example of Tensorboard outputs with the files provided in the root folder of this repository.

#### Predict
In order to predict you first have to train a model with any of the parameters. Then make sure to have the same parameters in the predict function (not that default parameters are common between `run.py` and `predict.py` but the parser is slightly different) and then run for example.

    python predict.py --split_by_phrase

This will use the provided example input file, but you can change it as well as choose the name of the generated .tsv file with the chords and the features if provided.
