# Speech emotion recognition with 2D CNN LSTM network in PyTorch

## Introduction
The network is similar to what Zhao et al. proposed in the paper [Speech emotion recognition using deep 1D & 2D CNN LSTM networks](https://www.sciencedirect.com/science/article/abs/pii/S1746809418302337).

## Table of contents
* [Datasets](#datasets)
  * [EMO-DB](#EMO-DB)
  * [EMOVO](#EMOVO)
* [Usage](#usage)
  * [Configuration](#configuration)
  * [Training](#training)
  * [Testing](#testing)
* [Acknowledgements](#Acknowledgements)
* [References](#References)

## Datasets

The configuration files are, by default, set to have the datasets in the `data` folder.

### EMO-DB
* [Paper](https://www.isca-speech.org/archive/archive_papers/interspeech_2005/i05_1517.pdf)
* [Download](http://www.emodb.bilderbar.info/download/)

### EMOVO
* [Paper](http://www.lrec-conf.org/proceedings/lrec2014/pdf/591_Paper.pdf)
* [Download](http://voice.fub.it/activities/corpora/emovo/index.html)

## Usage
The project structure is a rework of [victoresque](https://github.com/victoresque)'s PyTorch project template, so for more information check out his [repository](https://github.com/victoresque/pytorch-template).

### Configuration

### Training
A model can be trained by running `train.py` and passing the wanted configuration via the `--config` argument. E.g.:
```
python train.py --config <config file>.json
```

### Testing
A trained model can be tested by running `test.py` and passing the path to the saved checkpoint with the `--resume` argument.
For example, with the default configuration, it would be:
```
python test.py --resume saved/models/<model name>/<timestamp>/<checkpoint>.pth
```

## Acknowledgements
Thanks to [victoresque](https://github.com/victoresque) for the project template.

## References
Jianfeng Zhao, Xia Mao and Lijiang Chen. "[Speech emotion recognition using deep 1D & 2D CNN LSTM networks](https://www.sciencedirect.com/science/article/abs/pii/S1746809418302337)". in: Elsevier Biomedical Signal Processing and Control (2019), pp. 312–323.
