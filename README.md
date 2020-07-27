# TF2 CVRP solver with Multi Heads Attention 

<img src="https://user-images.githubusercontent.com/51239551/88506411-cd450f80-d014-11ea-84eb-12e7ab983780.gif" width="600"/>

TensorFlow implementation of ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!(https://arxiv.org/pdf/1803.08475.pdf)

## Desciption

[Slide Share -- CVRP solver with Multi Heads Attention --](https://www.slideshare.net/RINTAROSATO4/cvrp-solver-with-multi-head-attention?ref=https://www.slideshare.net/RINTAROSATO4/slideshelf)

## Dependencies

* Python >= 3.6
* TensorFlow = 2.1
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

