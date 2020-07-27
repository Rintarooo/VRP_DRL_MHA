# TF2 CVRP solver with Multi Heads Attention 

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

![VRP20 Pretrained](https://user-images.githubusercontent.com/51239551/88505529-adace780-d012-11ea-8247-efa9e604ff10.gif)


