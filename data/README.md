# Dataset & Configuration

## Introduction

### Dataset

Split a classification dataset, e.g., `Synthetic LEAF`, `MNIST` or `CIFAR-10`, among `n_clients`. 

Three splitting strategies are available:

#### IID split (Default)
The dataset is shuffled and partitioned among `n_clients`

#### By Labels Non-IID split
The dataset is split among `n_clients` as follows:
1. classes are grouped into `n_clusters`.
2. for each cluster `c` in `n_clusters`, samples are partitioned across clients using dirichlet distribution with parameter `alpha`.

Inspired by the split in [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440).

In order to use this mode, you should use argument `--by_labels_split`.

#### Pathological Non-IID split
The dataset is split as follow:
1) sort the data by label
2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
3) assign each of the `n_clients` with `n_classes_per_client` shards

Similar to [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

In order to use this mode, you should use argument `--pathological_split`.
   
### Configuration 

In order to simulate the heterogeneity of clients availability patterns
in realistic federated systems, we split the clients in two equally sized classes:
* ``available`` clients with probability $\pi = 1/2 + f$ of being active
* ``unavailable`` clients with  $\pi^{'} = 1/2 - f$ of being
  inactive, where $f_{i}\in(0, 1/2)$ is a parameter controlling
  the  heterogeneity of clients availability.
  
We furthermore split each class of clients in three equally sized sub-classes:
* ``stable`` clients that tend to keep the same activity state,
  with $\lambda^{(1)}=1-\varepsilon$
* ``unstable`` clients that tend to switch their activity state frequently,
  with $\lambda^{(2)}=-1+\varepsilon$
* ``shifting`` clients that are as likely to keep  as to change their activity state,
  with $\lambda^{(3)} \sim \mathcal{N}\left(0, \varepsilon^{2}\right)$ (default).

A deterministic split of clients into the different classes (e.g., clients are equally split among classes), 
is available using the argument `--deterministic_split`. To vary the proportion of clients in the sub-classes, 
please refer to the configuration file `JOINT_PROBABILITY_MATRIX` in `data/constants.py`.

## Instructions

Run `main.py` with a choice of the following arguments:

* ```--dataset_name```: dataset to use, possible are `"synthetic"`, `"mnist"` and `"cifar10"`
* ```--n_tasks```: (`int`) number of tasks/clients, given as an integer
* ```--dimension```: (`int`) dimension of the data, only used when `experiment_name=='synthetic'`
* ```--n_train_samples```: (`int`) number of train samples, only used when `experiment_name=='synthetic'`
* ```--n_test_samples```: (`int`) number of test samples, only used when `experiment_name=='synthetic'`
* ```--hetero_param```: (`float`) parameter controlling clients dissimilarity,
  only used when `experiment_name=='synthetic'`
  above is used
* ```----availability_parameters```: (`list`) parameters controlling the asymptotic
  availability of the clients from each group/cluster; 
  should be a list of the same size as `n_clusters`;
  default=`[0.0, 0.25]`
* ```--stability_parameters```: (`list`) list of parameters controlling the 
  stability of clients from each group/cluster; 
  should be a list of the same size as `n_clusters`;
  default=`[0.1, 0.25]`
* ```--save_dir```: (`str`) path of the directory to save data and configuration; 
  if not specified the data is saved to `./{dataset_name}`
* ```--seed```: (int) seed to be used to initialize the random number generator;
  if not provided, the system clock is used to generate the seed'


* ```--dataset_name```: Name of the dataset to use, possible values are {'synthetic', 'mnist', 'cifar10'}.
* ```--n_tasks```: (`int`) Number of tasks/clients.
* ```--frac```: Fraction of the dataset to be used. Default: 1.0.
* ```--n_classes```: Number of classes, only used when `experiment_name=='synthetic'`. Default: 10.
* ```--dimension```: Dimension of the data, only used when `experiment_name=='synthetic'`. Default: 60.
* ```--n_samples```: Total number of samples, only used when `experiment_name=='synthetic'`. Default: 10000.
* ```--alpha```: Parameter controlling how much local models differ from each other. Expected to be in the range (0,1). Default: 0.0.
* ```--beta```: Parameter controlling how much the local data at each device differs from that of other devices. Expected to be in the range (0,1). Default: 0.0.
* ```--iid```: If selected, data are split iid.
* ```--by_labels_split```: If selected, data are split non-iid (by labels). Only used when `experiment_name=='mnist'` or `experiment_name=='cifar10'`.
* ```--n_components```: Number of components/clusters. Ignored if `--by_labels_split` is not used. If `n_components=-1`, then `n_components` will be set to be equal to `n_classes(=10)`. Default: -1. 
* ```--dirichlet_alpha```: Parameter controlling clients dissimilarity. The smaller alpha is, the more clients are dissimilar. Ignored if `--by_labels_split` is not used. Default: 0.5.
* ```--pathological_split```: If selected, data are split non-iid (pathological). In case `--pathological_split` and `--by_labels_split` are both selected, `--by_ labels_split` will be used.
* ```--n_shards```: Number of shards given to each client. Ignored if `--pathological_split` is not used. Default: 1.
* ```--availability_parameter```: Parameter controlling the availability of the clients. Default: 0.4.
* ```--save_dir```: Path of the directory to save data and configuration. The directory will be created if not already created. If not specified, the data is saved to "./{dataset_name}".
* ```--seed```: (`int`) Seed for the random number generator. If not specified, the system clock is used to generate the seed.
  
## Paper Experiments

### Synthetic 

In order to generate the data split and configuration for the synthetic dataset experiment, run

```
python main.py \
  --dataset synthetic \
  --n_tasks 10 \
  --n_classes 10 \
  --n_samples 10000 \
  --dimension 60 \
  --alpha 1.0 \
  --beta 1.0 \
  --availability_parameter 0.4 \
  --stability_parameter 0.0 \
  --save_dir synthetic \
  --seed 123
```

### MNIST

In order to generate the data split and configuration for MNIST experiment, run

```
python main.py \
  --dataset mnist \
  --frac 0.1 \
  --n_tasks 10 \
  --by_labels_split \
  --n_components -1 \
  --alpha 0.3 \
  --availability_parameter 0.4 \
  --stability_parameter 0.0 \
  --save_dir mnist \
  --seed 123
```

### CIFAR-10

In order to generate the data split and configuration for CIFAR-10 experiment, run

```
python main.py \
  --dataset cifar10 \
  --frac 0.1 \
  --n_tasks 10 \
  --by_labels_split \
  --n_components -1 \
  --alpha 0.3 \
  --availability_parameter 0.4 \
  --stability_parameter 0.0 \
  --save_dir cifar10 \
  --seed 123
```