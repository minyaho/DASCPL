# Dynamic Accumulated Supervised Contrastive Parallel Learning
## Environment
| Name | Version | Note |
| --- | --- | --- |
| Python | `3.8.12` | Please install it from [Anaconda](https://www.anaconda.com/products/distribution). |
| CUDA | `11.4.1` | You can download from [here](https://developer.nvidia.com/cuda-11-4-1-download-archive). |
| PyTorch | `1.12.1+cu113` | Include `0.13.1+cu113` version of torchvision. </br> You can download from [here](https://pytorch.org/get-started/previous-versions/#v1121). |
||| Others in the `requirements.txt` file. </br> Please use pip to install them. |

## Setup
### Make a Environment

Tested under Python 3.8.12 in Ubuntu 20.04.
Install the required packages by

```bash
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install -r requirements.txt
```

In addition, you can simulate the experiment by Docker,

```bash
$ docker pull nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
$ docker run --gpus all --name dascpl_env -p 19000:8888 --shm-size="10g" nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
$ docker start dascpl_env
$ docker exec -it dascpl_env /bin/bash
$ apt-get update -y && apt-get upgrade -y
$ wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && /bin/bash ~/anaconda.sh -b && rm ~/anaconda.sh && source /root/anaconda3/bin/activate && conda init
$ conda create --name dascpl python=3.8.12 -y && conda activate dascpl
$ git clone https://github.com/minyaho/DASCPL.git
$ cd DASCPL/
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install -r requirements.txt
$ conda activate dascpl # used before every time experiment
```

### Download Datasets

#### Vision
* Tiny-imagenet-200:  Download [here](https://drive.google.com/file/d/10wl7UjC47xuUZG5zdUwSwHP1tlV-Ubf7/view?usp=share_link). This zip file of tinyImageNet dataset processed under PyTorch ImageFolder format. 
  > Unzip the zip file, please use this command (`unzip tiny-imagenet-200.zip`).
  > Put the unzipped folder (`./tiny-imagenet-200`) in the root of your project.

#### NLP
* IMDB: Please download the dataset from [here](https://drive.google.com/file/d/1Z2iqiPKF5wYCgXR-Tc9ZnQqUFVkJvypA/view?usp=share_link).
  > Put this file (`IMDB_Dataset.csv`) in the root of your project.

### Download Word Embedding
* Glove
  ```bash
  # cd to the path of your project
  $ wget https://nlp.stanford.edu/data/glove.6B.zip
  $ unzip glove.6B.zip
  # "glove.6B.300d.txt" must be put in the root of the project
  ```

## Quick Start
There are many arguments that can be used in the code.
### Vision 
#### Usage
```bash
$ python train_vision.py [Options]
```
#### Options
| Name | Default | Description |
| -------- | -------- | -------- |
|`--model`|`VGG_BP_m`|Model name|
|`--dataset`|`cifar10`| Dataset name </br> Options: `cifar10`, `cifar100` or `tinyImageNet`|
|`--times`|`1`| Times of experiment|
|`--epochs`|`200`| Number of epochs for training|
|`--train_bsz`|`1024`| Batch size of training data|
|`--test_bsz`|`1024`| Batch size of test data|
|`--base_lr`|`0.001`| Initial learning rate|
|`--end_lr`|`0.00001`| Learning rate at the end of training|
|`--gpus`|`0`| ID of the GPU device. If you want to use multiple GPUs, you can separate them with commas, e.g., `0,1`. The model type is Single GPU will only use first gpu id.|
|`--seed`|`-1`| Random seed in the experiment. If you don't want to fix the random seed, you need to type `-1`|
|`--multi_t`|`true`| Multi-threaded on-off flag. On is "true". Off is "false"|
|`--proj_type`|`None`| Projective head type in contrastive loss. `i` is identity. `l` is linear. `m` is mlp.|
|`--pred_type`|`None`| Predictor type in predict loss. `i` is identity. `l` is linear. `m` is mlp.|
|`--save_path`|`None`| Save path of the model log. There are many types of logs, such as training logs, model results (JSON) and tensorboard files. "None" means do not save.|
|`--profiler`|`false`| Profiler of model. If you want to use the profiler, please type "true" and set the "save_path". "false" means do not use and save.|
|`--train_eval`|`ture`| On-off flag for evaluation behavior during training. |
|`--aug_type`|`strong`| Type of Data augmentation. Use **basic** augmentation like BP commonly used, or **strong** augmentation like contrastive learning used. </br> Options: `basic`, `strong` |

#### Model
* `VGG8`
  * SingleGPU: `VGG_BP`, `VGG_SCPL`
  * MultiGPU: `VGG_BP_m`, `VGG_BP_p_m`, `VGG_SCPL_m`, `VGG_DASCPL_m`
* `ResNet18`
  * SingleGPU: `resnet_BP`, `resnet_SCPL`
  * MultiGPU: `resnet_BP_m`, `resnet_BP_p_m`, `resnet18_SCPL_m`, `resnet_DASCPL_m`
* Suffix meaning
  * `m`: MultiGPU model. Similarly, it can also be experimented with a single GPU.
  * `p`: Can set a predictor in this model.  You need to set the `pred_type`. All DSCPL type models have this option by default (not shown in the suffix).

#### Dataset
`cifar10`, `cifar100` or `tinyImageNet`

#### Projector Type
This option is only available on **MultiGPU type** of SCPL or DASCPL.

#### Predictor Type
This option is only available on **MultiGPU type** of DASCPL or p-suffix models.

#### Example
```bash
$ python train_vision.py \
  --model="VGG_SCPL_m" --dataset="cifar10" --times=5  \
  --train_bsz=1024 --test_bsz=1024 \
  --base_lr=0.001 --end_lr=0.00001 \
  --epochs=200 --seed=-1 \
  --multi_t="true" --gpus="0" \
  --proj_type="m" --aug_type="strong"
```

### NLP
#### Usage
```bash
$ python train_nlp.py [Options]
```
#### Options
| Name | Default | Description |
| -------- | -------- | -------- |
|`--model`|`LSTM_BP_m_d`|Model name|
|`--dataset`|`ag_news`| Dataset name </br> Options: `ag_news`, `dbpedia_14`, `sst2`, `imdb`|
|`--times`|`1`| Times of experiment|
|`--epochs`|`50`| Number of epochs for training|
|`--train_bsz`|`1024`| Batch size of training data|
|`--test_bsz`|`1024`| Batch size of test data|
|`--base_lr`|`0.001`| Initial learning rate|
|`--end_lr`|`0.001`| Learning rate at the end of training|
|`--gpus`|`0`| ID of the GPU device. If you want to use multiple GPUs, you can separate them with commas, e.g., `0,1`. The model type is Single GPU will only use first gpu id.|
|`--seed`|`-1`| Random seed in the experiment. If you don't want to fix the random seed, you need to type `-1`|
|`--multi_t`|`true`| Multi-threaded on-off flag. On is "true". Off is "false"|
|`--proj_type`|`None`| Projective head type in contrastive loss. `i` is identity. `l` is linear. `m` is mlp.|
|`--pred_type`|`None`| Predictor type in predict loss. `i` is identity. `l` is linear. `m` is mlp.|
|`--save_path`|`None`| Save path of the model log. There are many types of logs, such as training logs, model results (JSON) and tensorboard files. "None" means do not save.|
|`--profiler`|`false`| Profiler of model. If you want to use the profiler, please type "true" and set the "save_path". "false" means do not use and save.|
|`--train_eval`|`ture`| On-off flag for evaluation behavior during training |
|`--max_len`|`60`| Maximum length for the sequence of input samples |
|`--h_dim`|`300`|Dimensions of the hidden layer|
|`--layers`|`4`|Number of layers of the model. The minimum is `2`. Because the first layer is the pre-training embedding layer, and the latter layer is lstm or transformer.|
|`--heads`|`6`|Number of heads of transformer encoder. This option is only available on transformer.|
|`--vocab_size`|`30000`|Size of dictionary vocabulary|
|`--word_vec`|`glove`|Type of word embedding|
|`--emb_dim`|`300`|Dimension of word embedding|

#### Model
* `LSTM`
  * SingleGPU: `LSTM_BP_3`, `LSTM_BP_4`, `LSTM_BP_d`, `LSTM_SCPL_3`, `LSTM_SCPL_4`
  * MultiGPU: `LSTM_BP_m_d`, `LSTM_BP_p_m_d`, `LSTM_SCPL_m_d`, `LSTM_DASCPL_m_d`
* `Transformer`
  * SingleGPU: `Trans_BP_3`, `Trans_BP_4`, `Trans_BP_d`, `Trans_SCPL_3`, `Trans_SCPL_4` 
  * MultiGPU: `Trans_BP_m_d`, `Trans_BP_p_m_d`, `Trans_SCPL_m_d`, `Trans_DASCPL_m_d`
* Suffix meaning
  * `<number>`: The number of layers.
    e.g., The `LSTM_SCPL_3` model has three layers in this model.
  * `m`: MultiGPU model. Similarly, it can also be experimented with a single GPU.
  * `d`: Customize the number of layers.
  * `p`: Can set a predictor in this model.  You need to set the `pred_type`. All DSCPL type models have this option by default (not shown in the suffix).
#### Dataset


| Name | max_len |
| ---- | ------- |
| `sst2` | 15 |
| `ag_news` | 60 |
| `imdb`| 350 |
| `dbpedia_14` | 400 |

#### Projector Type
This option is only available on **MultiGPU type** of SCPL or DASCPL.

#### Predictor Type
This option is only available on **MultiGPU type** of DASCPL or p-suffix models.

#### Example
```bash
$ python train_nlp.py \
  --model="LSTM_SCPL_m_d" --dataset="ag_news" --times=5  \
  --train_bsz=1024 --test_bsz=1024 \
  --base_lr=0.001 --end_lr=0.001 \
  --epochs=50 --seed=-1 \
  --multi_t="true" --gpus="0" \
  --proj_type="i" --max_len=60 \
  --h_dim=300 --layers=4
```