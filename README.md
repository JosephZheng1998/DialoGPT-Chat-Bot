# DialoGPT-Chat-Bot
This repo contains the DialoGPT Models for CMU 11747 NN for NLP. The code and README is based on the [repo](https://github.com/huggingface/transfer-learning-conv-ai) accompany the blog post [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/@Thomwolf/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313).

## Installation

To install and use the training and inference scripts please clone the repo and install the requirements:

```bash
git clone https://github.com/JosephZheng1998/DialoGPT-Chat-Bot.git
cd DialoGPT-Chat-Bot
pip3 install -r requirements.txt
python3 -m spacy download en
```

## Installation with Docker

To install using docker please build the self-contained image:

```bash
docker build -t convai .
```

_Note: Make sure your Docker setup allocates enough memory to building the container. Building with the default of 1.75GB will fail due to large Pytorch wheel._

You can then enter the image  

```bash
ip-192-168-22-157:transfer-learning-conv-ai loretoparisi$ docker run --rm -it convai bash
root@91e241bb823e:/# ls
Dockerfile  README.md  boot                  dev  home         lib    media  models  proc              root  sbin  sys  train.py  utils.py
LICENCE     bin        convai_evaluation.py  etc  interact.py  lib64  mnt    opt     requirements.txt  run   srv   tmp  usr       var
```

You can then run the `interact.py` script on the pretrained model:

```bash
python3 interact.py --model models/
```

## Using the training script

The training script can be used in single GPU or multi GPU settings:

```bash
python3 ./train.py  # Single GPU training
python3 -m torch.distributed.launch --nproc_per_node=8 ./train.py  # Training on 8 GPUs
```

The training script accept several arguments to tweak the training:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset. If empty download from S3.
dataset_cache | `str` | `'./dataset_cache.bin'` | Path or url of the dataset cache
model | `str` | `"openai-gpt"` | Path, url or short name of the model
num_candidates | `int` | `2` | Number of candidates for training
max_history | `int` | `2` | Number of previous exchanges to keep in history
train_batch_size | `int` | `4` | Batch size for training
valid_batch_size | `int` | `4` | Batch size for validation
gradient_accumulation_steps | `int` | `8` | Accumulate gradients on several steps
lr | `float` | `6.25e-5` | Learning rate
lm_coef | `float` | `1.0` | LM loss coefficient
mc_coef | `float` | `1.0` | Multiple-choice loss coefficient
max_norm | `float` | `1.0` | Clipping gradient norm
n_epochs | `int` | `3` | Number of training epochs
personality_permutations | `int` | `1` | Number of permutations of personality sentences
device | `str` | `"cuda" if torch.cuda.is_available() else "cpu"` | Device (cuda or cpu)
fp16 | `str` | `""` | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)
local_rank | `int` | `-1` | Local rank for distributed training (-1: not distributed)

Here is how to reproduce our results on a server with 8 A100 GPUs (adapt number of nodes and batch sizes to your configuration):

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 ./train.py --model_checkpoint microsoft/DialoGPT-medium --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=3 --num_candidates=4 --personality_permutations=2 --train_batch_size=4 --valid_batch_size=4
```

## Running ConvAI2 evaluation scripts

To run the evaluation scripts of the ConvAI2 challenge, you first need to install `ParlAI` in the repo base folder like this:

```bash
git clone https://github.com/facebookresearch/ParlAI.git
cd ParlAI
python setup.py develop
```

You can then run the evaluation script from `ParlAI` base folder:

```bash
cd ParlAI
python ../convai_evaluation.py --eval_type hits@1  # to download and evaluate our fine-tuned model on hits@1 metric
python ../convai_evaluation.py --eval_type hits@1  --model_checkpoint ./data/Apr17_13-31-38_thunder/  # to evaluate a training checkpoint on hits@1 metric
```

The evaluation script accept a few arguments to select the evaluation metric and tweak the decoding algorithm:

Argument | Type | Default value | Description
---------|------|---------------|------------
eval_type | `str` | `"hits@1"` | Evaluate the model on `hits@1`, `ppl` or `f1` metric on the ConvAI2 validation dataset
model | `str` | `"openai-gpt"` | Path, url or short name of the model
max_history | `int` | `2` | Number of previous utterances to keep in history
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `20` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `42` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

## Data Format
see `example_entry.py`, and the comment at the top.

## Reference
- [transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)
