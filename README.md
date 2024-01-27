# Projet SAM

Our project focuses on multimodal approaches for predicting turn-taking changes in natural conversations. The goals of this project enable us to introduce various concepts of textual, visual, and auditory modality, as well as to compare and explore different multimodal processing models and their fusion.

- [Projet SAM](#projet-sam)
- [Requirements](#requirements)
- [Launch the code](#launch-the-code)
  - [Mode](#mode)
  - [Configuration](#configuration)
  - [Help](#help)
  - [Example](#example)
- [Models](#models)
  - [Unimodal Models](#unimodal-models)
    - [TEXT](#text)
    - [AUDIO](#audio)
    - [VIDEO](#video)
  - [Multimodal Models](#multimodal-models)
    - [LATE FUSION](#late-fusion)
    - [EARLY FUSION](#early-fusion)
- [Results](#results)


# Requirements
To run the code you need python (We use python 3.9.13) and packages that is indicate in [`requirements.txt`](requirements.txt).
You can run the following code to install all packages in the correct versions:
```sh
pip install -r requirements.txt
```

# Launch the code

The [`main.py`](main.py) script is the main entry point for this project. It accepts several command-line arguments to control its behavior:

- `--mode` or `-m`: This option allows you to choose a mode between 'train', 'test' and 'latefusion'.
- `--config_path` or `-c`: This option allows you to specify the path to the configuration file. The default is [`config/config.yaml`](config/config.yaml). Use only for the training.
- `--path` or `-p`: This option allows you to specify the experiment path for testing.
- `--task` or `-t`: This option allows you to specify the task for the model. This will overwrite the task specified in the configuration file for training. It's can be `text`, `audio`, `video`, or `multi`, that will train a new experiments with this type of data. The task multi will be use the unimodal models that is indicate in the parameter 'load' in the configuration file.

## Mode
Here's what each mode does:

- [`train`](src/train.py): Trains a model using the configuration specified in the `--config_path` and the task specified in `--task`.
- [`test`](src/test.py): Tests the model specified in the `--path`. You must specify a path.
- `latefusion`: test a late fusion with models which is in load in the configuration file. There's no need to train this model, as it has no learnable parameters.

## Configuration

The configuration of the model and the training process is done through a YAML file. You can specify the path to this file with the `--config_path` option. The default path is [`config/config.yaml`](config/config.yaml).

The configuration file includes various parameters such as the learning rate, batch size, number of epochs, etc.

## Help

To get a list of all available options, you can use the `-h` or `--help` option:

```sh
python main.py --help
```

This will display a help message with a description of all available options.

## Example
Here's an example of how to use the script to train a model:

```sh
python main.py --mode train --config_path config/config.yaml --task text
```

This command will train a model using the configuration specified in [`config/config.yaml`](config/config.yaml) with a `task=text`.

Here's an example of how to run a test on the experiment separete:

```sh
python main.py --mode test --path logs/multi_4
```

# Models
## Unimodal Models
### TEXT
<p align="center"><img src=report\image_model\model_text.png><p>

### AUDIO
<p align="center"><img src=report\image_model\model_audio.png><p>

### VIDEO
<p align="center"><img src=report\image_model\model_video.png><p>

## Multimodal Models
### LATE FUSION
<p align="center"><img src=report\image_model\late_fusion.png><p>

### EARLY FUSION
<p align="center"><img src=report\image_model\early_fusion.png><p>

# Results

Model test results. The *LATE* and *EARLY FUSION* do not use the *VIDEO* model.

| Models            | Accuracy | Precision | Recall | $f_1$ score |
|-------------------|----------|-----------|--------|-------------|
| *TEXT*            | 82.8     | 41.3      | 50.0   | 45.3        |
| *AUDIO*           | 47.1     | 48.5      | 47.4   | 41.5        |
| *VIDEO*           | **82.9** | 41.4      | 50.0   | 45.2        |
| *LATE FUSION*     | 78.5     | **50.6**  | 50.1   | **48.8**    |
| *EARLY FUSION*    | **82.9** | 43.6      | **50.2** | 45.7      |

