# CVRP solver with Multi Heads Attention in TF2, Torch

<img src="https://user-images.githubusercontent.com/51239551/88506411-cd450f80-d014-11ea-84eb-12e7ab983780.gif" width="650"/>

<img src="https://user-images.githubusercontent.com/51239551/88507610-bfdd5480-d017-11ea-99de-e9850e6be0db.gif" width="650"/>

<img src="https://user-images.githubusercontent.com/51239551/89150677-0ee83400-d59a-11ea-90ed-2852dc1ddd4b.gif" width="650"/>

TensorFlow and PyTorch implementation of ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!(https://arxiv.org/pdf/1803.08475.pdf)

## Description

[Slide Share -- CVRP solver with Multi Heads Attention --](https://www.slideshare.net/RINTAROSATO4/cvrp-solver-with-multihead-attention)


## Dependencies

* Python >= 3.6
* TensorFlow >= 2.0
* PyTorch = 1.5
* tqdm
* scipy
* numpy
* plotly (only for plotting)
* matplotlib (only for plotting)


## Usage

First generate the pickle file contaning hyperparameter values by running the following command.

```
python config.py
```

then, train the model.

```
python train.py -p './Pkl/***.pkl'
```

Plot prediction of the pretrained model

```
python plot.py -p './Weights/***.h5'
```

