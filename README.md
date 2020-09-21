# CVRP solver with Multi Heads Attention

TensorFlow2 and PyTorch implementation of ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!(Kool et al. 2019)(https://arxiv.org/pdf/1803.08475.pdf)

<img src="https://user-images.githubusercontent.com/51239551/88506411-cd450f80-d014-11ea-84eb-12e7ab983780.gif" width="650"/>

<img src="https://user-images.githubusercontent.com/51239551/88507610-bfdd5480-d017-11ea-99de-e9850e6be0db.gif" width="650"/>

<img src="https://user-images.githubusercontent.com/51239551/89150677-0ee83400-d59a-11ea-90ed-2852dc1ddd4b.gif" width="650"/>

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

First move to "PyTorch" or "TensorFlow2" dir. 

```
cd PyTorch
```

Then, generate the pickle file contaning hyperparameter values by running the following command.

```
python config.py
```

you would see the pickle file in "Pkl" dir. now you can start training the model.

```
python train.py -p Pkl/***.pkl
```

Plot prediction of the pretrained model

```
python plot.py -p Weights/***.pt(or ***.h5)
```

If you want to verify your model, you can use opensource dataset in "OpenData" dir.
  
Opensource data is obtained from Augerat et al.(1995)
  
please refer to [Capacitated VRP Instances by NEO Research Group](https://neo.lcc.uma.es/vrp/vrp-instances/capacitated-vrp-instances/)
```
python plot.py -p Weights/***.pt -t ../OpenData/A-n***.txt
```

## Reference
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/d-eremeev/ADM-VRP
* https://qiita.com/ohtaman/items/0c383da89516d03c3ac0