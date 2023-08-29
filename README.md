# Dynamic Accumulated Supervised Contrastive Parallel Learning

Supervised Contrastive Parallel Learning (SCPL) is a novel approach that decouples BP by multiple local training objectives and supervised contrastive learning. It transforms the original deep network's long gradient flow into multiple short gradient flows and trains the parameters in different layers independently through a pipelined design. This method achieves faster training speed than BP by addressing the inefficiency caused by backward locking in backpropagation. 

We improved the architecture of SCPL, which enables dynamic layer accumulation, forward shortcuts, and early exits. This new architecture is called Dynamic Accumulated Supervised Contrastive Parallel Learning (DASCPL). Based on these two features, DASCPL offers higher flexibility and adaptability compared to SCPL while maintaining consistent learning capabilities.

Now, both DASCPL and SCPL can be demonstrated in the visual and natural language domains through this repository.

## Environment
| Name | Version | Note |
| --- | --- | --- |
| Python | `3.8.12` | Please install it from [Anaconda](https://www.anaconda.com/products/distribution). |
| CUDA | `11.4.1` | You can download it from [here](https://developer.nvidia.com/cuda-11-4-1-download-archive). |
| PyTorch | `1.12.1+cu113` | Include `0.13.1+cu113` version of torchvision. </br> You can download it from [here](https://pytorch.org/get-started/previous-versions/#v1121). |
||| Others in the `requirements.txt` file. </br> Please use pip to install them. |

The packages listed above are the ones we use in our development environment. However, this environment may encounter some issues during testing. To address this, we have provided an alternative list of environments which we have tried as follows:

| Name | Version | Note |
| --- | --- | --- |
| Python | `3.8.12` | Please install it from [Anaconda](https://www.anaconda.com/products/distribution). |
| CUDA | `12.0.1` | You can download it from [here](https://developer.nvidia.com/cuda-12-0-1-download-archive). |
| PyTorch | `2.0.1+cu118` | Include `0.15.2+cu118` version of torchvision. </br> You can download it from [here](https://pytorch.org/get-started/previous-versions/#v201) or [here](https://pytorch.org/). |
||| Others in the `requirements.txt` file. </br> Please use pip to install them. |

## Setup
### Make an Environment
#### General use
Tested under Python 3.8.12 on Ubuntu 20.04.
Install the required packages by running the following command:

```bash
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install -r requirements.txt
```

We also have provided an alternative list of environments which we have tried as follows:

```bash
$ pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```

#### Docker use 
Additionally, you can simulate the experiment using Docker with the following steps:

```bash
# Install Env
$ docker pull nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
$ docker run --gpus all --name dascpl_env -it -p 19000:8888 --shm-size="10g" nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
$ # If "nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04" is not available, use "minyaho/cuda:11.4.1-cudnn8-devel-ubuntu20.04
$ apt-get update -y 
$ apt-get upgrade -y
$ apt-get install git wget -y
$ wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && /bin/bash ~/anaconda.sh -b && rm ~/anaconda.sh && source /root/anaconda3/bin/activate && conda init
$ conda create --name dascpl python=3.8.12 -y && conda activate dascpl
$ cd ~ && git clone https://github.com/minyaho/dascpl.git
$ cd ~/dascpl/
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install -r requirements.txt
$ python -m ipykernel install --user --name dascpl --display-name "dascpl"
$ exit
$ docker stop dascpl_env

# Used before every time experiment
$ docker start dascpl_env
$ docker exec -it dascpl_env /bin/bash
$ cd ~/dascpl/ && conda activate dascpl 
$ # pip install notebook==6.4.8 # If you want to use Jupyter Notebook.
$ # jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="scpl"# If you want to use Jupyter Notebook, please execute this command and access it through port 19000 and token is "scpl". 
```

We also have provided an alternative list of environments which we have tried as follows:

```bash
# Install Env
$ docker pull nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
$ docker run --gpus all --name dascpl_env -it -p 19000:8888 --shm-size="10g" nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
$ # If "nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04" is not available, use "minyaho/cuda:12.0.1-cudnn8-devel-ubuntu20.04"
$ apt-get update -y 
$ apt-get upgrade -y
$ apt-get install git wget -y
$ wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && /bin/bash ~/anaconda.sh -b && rm ~/anaconda.sh && source /root/anaconda3/bin/activate && conda init
$ conda create --name dascpl python=3.8.12 -y && conda activate dascpl
$ cd ~ && git clone https://github.com/minyaho/dascpl.git
$ cd ~/dascpl/
$ pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
$ python -m ipykernel install --user --name dascpl --display-name "dascpl"
$ exit
$ docker stop dascpl_env

# Used before every time experiment
$ docker start dascpl_env
$ docker exec -it dascpl_env /bin/bash
$ cd ~/dascpl/ && conda activate dascpl # used before every time experiment
$ # pip install notebook==6.4.8 # If you want to use Jupyter Notebook.
$ # jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="scpl"# If you want to use Jupyter Notebook, please execute this command and access it through port 19000 and token is "scpl". 
```

If you don't want to create the environment yourself, you can also directly get a pre-prepared image from Docker Hub.

- CUDA `11.4.1` and Pytorch `1.12.1+cu113`

```bash
# Install Env
$ docker pull minyaho/dascpl:c1141p1121
$ docker run --gpus all --name dascpl_env -it -p 19000:8888 --shm-size="10g" minyaho/dascpl:c1141p1121
$ cd ~ && git clone https://github.com/minyaho/dascpl.git
$ exit
$ docker stop dascpl_env

# Used before every time experiment
$ docker start dascpl_env
$ docker exec -it dascpl_env /bin/bash
$ cd ~/dascpl/ && conda activate dascpl
$ # pip install notebook==6.4.8 # If you want to use Jupyter Notebook.
$ # jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="scpl"# If you want to use Jupyter Notebook, please execute this command and access it through port 19000 and token is "scpl". 
```

- CUDA `12.0.1` and Pytorch `2.0.1+cu118`
```bash
# Install Env
$ docker pull minyaho/dascpl:c1201p201
$ docker run --gpus all --name dascpl_env -it -p 19000:8888 --shm-size="10g" minyaho/dascpl:c1201p201
$ cd ~ && git clone https://github.com/minyaho/dascpl.git
$ exit
$ docker stop dascpl_env

# Used before every time experiment
$ docker start dascpl_env
$ docker exec -it dascpl_env /bin/bash
$ cd ~/dascpl/ && conda activate dascpl
$ # pip install notebook==6.4.8 # If you want to use Jupyter Notebook.
$ # jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="scpl"# If you want to use Jupyter Notebook, please execute this command and access it through port 19000 and token is "scpl". 
```

### Download Datasets

#### Vision
* Tiny-imagenet-200:  Download [here](https://drive.google.com/file/d/10wl7UjC47xuUZG5zdUwSwHP1tlV-Ubf7/view?usp=share_link). This zip file contains the tinyImageNet dataset processed in the PyTorch ImageFolder format.
  > Unzip the file using the command  `unzip tiny-imagenet-200.zip`.
  > Place the unzipped folder (`./tiny-imagenet-200`) in the root of your project.

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
|`--times`|`1`| Number of experiments to run|
|`--epochs`|`200`| Number of training epochs|
|`--train_bsz`|`1024`| Batch size of training data|
|`--test_bsz`|`1024`| Batch size of test data|
|`--base_lr`|`0.001`| Initial learning rate|
|`--end_lr`|`0.00001`| Learning rate at the end of training|
|`--temperature`|`0.1`|Temperature parameter of contrastive loss|
|`--gpus`|`0`| ID of the GPU device. If you want to use multiple GPUs, you can separate their IDs with commas, e.g., `0,1`. For single GPU models, only the first GPU ID will be used.|
|`--seed`|`-1`| Random seed used in the experiment. Use `-1` to generate a random seed for each run.|
|`--multi_t`|`true`| Multi-threading flag. Set it to "true" to enable multi-threading, or "false" to disable it.|
|`--proj_type`|`None`| Projective head type in contrastive loss. Use `i` for identity, `l` for linear, and `m` is mlp.|
|`--pred_type`|`None`| Predictor type in predict loss. Use `i` for identity, `l` for linear, and `m` is mlp.|
|`--save_path`|`None`| Save path of the model log. Different types of logs, such as training logs, model results (JSON), and tensorboard files, can be saved. Use "None" to disable saving.|
|`--profiler`|`false`| Model profiler. Set it to "true" to enable the profiler and specify the "save_path". Set it to "false" to disable the profiler.|
|`--train_eval`|`ture`|Flag to enable evaluation during training (only for multi-GPU models). |
|`--train_eval_times`|`1`| The number of epochs between evaluations during training.|
|`--temperature`|`0.1`| Temperature parameter of contrastive loss. |
|`--aug_type`|`strong`| Type of Data augmentation. Use **basic** augmentation like BP (backpropagation) commonly used, or **strong** augmentation like contrastive learning used. </br> Options: `basic`, `strong` |

#### Model
* `VGG8`
  * SingleGPU: `VGG_BP`, `VGG_SCPL`
  * MultiGPU: `VGG_BP_m`, `VGG_BP_p_m`, `VGG_SCPL_m`, `VGG_DASCPL_m`
* `ResNet18`
  * SingleGPU: `resnet_BP`, `resnet_SCPL`
  * MultiGPU: `resnet_BP_m`, `resnet_BP_p_m`, `resnet18_SCPL_m`, `resnet_DASCPL_m`
* Suffix meaning
  * `m`: MultiGPU model. Similarly, it can also be experimented with a single GPU.
  * `p`: Specify the predictor in a model.  You need to set the `pred_type`. All DSCPL type models have this option by default (not shown in the suffix).

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
  --proj_type="m" --aug_type="strong" \
  --temperature=0.1
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
|`--times`|`1`| Number of experiments to run|
|`--epochs`|`50`| Number of training epochs|
|`--train_bsz`|`1024`| Batch size of training data|
|`--test_bsz`|`1024`| Batch size of test data|
|`--base_lr`|`0.001`| Initial learning rate|
|`--end_lr`|`0.001`| Learning rate at the end of training|
|`--temperature`|`0.1`|Temperature parameter of contrastive loss|
|`--gpus`|`0`| ID of the GPU device. If you want to use multiple GPUs, you can separate them with commas, e.g., `0,1`. For single GPU models, only the first GPU ID will be used.|
|`--seed`|`-1`| Random seed in the experiment. Use `-1` to generate a random seed for each run.|
|`--multi_t`|`true`| Multi-threading flag. Set it to "true" to enable multi-threading, or "false" to disable it.|
|`--proj_type`|`None`| Projective head type in contrastive loss. Use `i` for identity, `l` for linear, and `m` for mlp.|
|`--pred_type`|`None`| Predictor type in predict loss. Use `i` for identity, `l` for linear, and `m` for mlp.|
|`--save_path`|`None`| Save path of the model log. Different types of logs, such as training logs, model results (JSON), and tensorboard files, can be saved. Use "None" to disable saving.|
|`--profiler`|`false`| Model profiler. Set it to "true" to enable the profiler and specify the "save_path". Set it to "false" to disable the profiler. |
|`--train_eval`|`ture`| Flag to enable evaluation during training (only for multi-GPU models). |
|`--train_eval_times`|`1`| The number of epochs between evaluations during training.|
|`--temperature`|`0.1`| Temperature parameter of contrastive loss. |
|`--max_len`|`60`| Maximum length for the sequence of input samples |
|`--h_dim`|`300`|Dimensions of the hidden layer|
|`--layers`|`4`|Number of layers of the model. The minimum is `2`. The first layer is the pre-training embedding layer, and the latter layer is lstm or transformer.|
|`--heads`|`6`|Number of heads in the transformer encoder. This option is only available for the Transformer model.|
|`--vocab_size`|`30000`|Size of vocabulary dictionary.|
|`--word_vec`|`glove`| Type of word embedding |
|`--emb_dim`|`300`| Dimension of word embedding |
|`--noise_rate`|`0.0`| Noise rate of labels in training dataset (default is 0 for no noise). |

#### Model
* `LSTM`
  * SingleGPU: `LSTM_BP_3`, `LSTM_BP_4`, `LSTM_BP_d`, `LSTM_SCPL_3`, `LSTM_SCPL_4`
  * MultiGPU: `LSTM_BP_m_d`, `LSTM_BP_p_m_d`, `LSTM_SCPL_m_d`, `LSTM_DASCPL_m_d`
* `Transformer`
  * SingleGPU: `Trans_BP_3`, `Trans_BP_4`, `Trans_BP_d`, `Trans_SCPL_3`, `Trans_SCPL_4` 
  * MultiGPU: `Trans_BP_m_d`, `Trans_BP_p_m_d`, `Trans_SCPL_m_d`, `Trans_DASCPL_m_d`
* Suffix meaning
  * `<number>`: The number of layers.
    e.g., the model `LSTM_SCPL_3` has three layers.
  * `m`: MultiGPU model. Similarly, it can also be experimented with a single GPU.
  * `d`: Customize the number of layers.
  * `p`: Specify the predictor in a model.  You need to set the `pred_type`. All DSCPL type models have this option by default (not shown in the suffix).
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
  --h_dim=300 --layers=4 \
  --temperature=0.1
```
