# README for TACAS 2025 Artifact Evaluation

This README is specifically for the [artifact evaluation](https://tacas.info/artifacts-25.php).
Another [README for general users](https://github.com/shizhouxing/GenBaB) is available in the code repository.

## Abstract

This artifact contains the implementation for the **GenBaB** algorithm proposed in our paper and the benchmarks used in our experiments. The artifact can be used to reproduce the experiments in our paper and is expected to be reusable for new models and specifications by future works.

## Links

Link to the Zenodo repository: **LINK TO BE ADDED**

Our code is also hosted on [GitHub](https://github.com/shizhouxing/GenBaB) and our benchmarks are hosted on [HuggingFace](https://github.com/shizhouxing/GenBaB).
GenBaB is implemented into the [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) toolbox, and a new version of α,β-CROWN with GenBaB integrated will be released to the main [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) repository.

The Zenodo repository contains a current copy of the code on GitHub, as well as the benchmarks. For the artifact evaluation, you may directly use the version from Zenodo.
For readability, this README for artifact evaluation contains some links to the [README for general users](https://github.com/shizhouxing/GenBaB) in the code repository hosted on GitHub, but that README is also available on Zenodo.

## Requirements

**Special requirements: An NVIDIA GPU compatible with PyTorch 2.2 is required.**
Instead of using the TACAS VM without GPU support, we kindly request the reviewer to use a GPU server with Linux and CUDA>=11.8.

If you wish to try GenBaB on the TACAS VM with CPU only, an [instruction](https://github.com/shizhouxing/GenBaB/blob/main/README_tacas.md#trying-genbab-on-a-cpu-only-machine) is attached at the end of this README.

## Setup

### Installing Python 3.11

Python 3.11+ is required. We recommend using [conda](https://docs.anaconda.com/miniconda/) to setup a clean Python environment. Follow the [installation guide](https://docs.anaconda.com/miniconda/install/) to install miniconda.

If you are using a Linux x86 environment, you may install miniconda to `~/miniconda3` by:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
bash
```

Create and activate a new environment with Python 3.11 for GenBaB:
```bash
conda create --name GenBaB -y python=3.11
conda activate GenBaB
```

### Installing PyTorch 2.2

Use `conda` to install PyTorch 2.2 compatible with your CUDA version:
```
# If you are using CUDA 11.8
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# if you are using CUDA 12.1 or above
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Installing other dependencies

After setting up Python and PyTorch, install other dependencies by:
```bash
cd alpha-beta-CROWN
pip install -e .
cd complete_verifier
pip install -r requirements.txt
```

## Usage

See [the "Usage" section](https://github.com/shizhouxing/GenBaB?tab=readme-ov-file#usage) in the README of the code for the general usage of GenBaB.

### Light review

To perform a light review, you may try running GenBaB
on the first 5 instances of the CIFAR-10 Sigmoid 4x100 model:
```
# Please run the code from the `alpha-beta-CROWN/complete_verifier` directory
# (if you are not at this directory yet)
cd alpha-beta-CROWN/complete_verifier

# `--end 5` means that we only try the first 5 instances instead of the entire benchmark
python abcrown.py --config ../../benchmarks/cifar/sigmoid_4fc_100/config.yaml --end 5
```

## Full review

The major required resource is an NVIDIA GPU compatible with PyTorch 2.2.
We used a single NVIDIA GTX 1080 Ti GPU for each of the experiments.
More advanced GPU models compatible with PyTorch 2.2 are expected to work better.

First, run [`warmup.sh`](./warmup.sh) which will run each kind of model on a single instance with a short timeout to build the lookup table of pre-optimized branching points.
The warmup script will run each kind of model on a single instance with a short timeout to build the lookup table of pre-optimized branching points. Since this lookup table can be shared by all the instances with existing model architectures, the warmup step can separate the time cost of building the lookup table from the main experiments. Otherwise, the cost of pre-optimizing branching points may be counted toward the first instance of each new model architecture.

Script [`run.sh`](./run.sh) contains a list of commands for running GenBaB on all the experimented benchmarks. Please inspect the script to understand the experiments involved, and you may select a subset of them to run for the evaluation.

If you want to run variants of GenBaB or the baseline without branch-and-bound, you may add options:
* `--complete_verifier skip`: Disable branch-and-bound.
* `--nonlinear_split_method babsr-like`: Use a BaBSR-like branching heuristic instead of BBPS proposed in the paper.
* `--branching_point_method uniform`: Disable optimized branching points.
* `--nonlinear_split_relu_only`: For models with a mix of ReLU and other nonlinearities, only consider branching ReLU neurons, not other nonlinearities.

## GenBaB is expected to be "Reusable"

As documented in the README of the code, GenBaB can [take new models and specifications](https://github.com/shizhouxing/GenBaB?tab=readme-ov-file#running-genbab-on-new-models) defined in VNN-COMP format.
Therefore, GenBaB is expected to be "Reusable".

## Trying GenBaB on a CPU-only machine

You may try GenBaB on a CPU-only machine, but a GPU machine is necessary to reproduce the regular performance of GenBaB. We now assume that you are using the [TACAS 23 Artifact Evaluation VM](https://zenodo.org/records/7113223) with Ubuntu 22.04 or a similar environment.

Please configure the VM to enable internet access which is required for installing external dependencies as mentioned above.

Install conda and create a virtual environment with Python 3.11 for GenBaB (in the same way as mentioned earlier for GPU machines):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
bash
conda create --name GenBaB -y python=3.11
conda activate GenBaB
```

Install the *CPU-version* PyTorch 2.2.0:
```bash
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```

Install other dependencies by (in the same way as mentioned earlier for GPU machines):
```bash
cd alpha-beta-CROWN
pip install -e .
cd complete_verifier
pip install -r requirements.txt
```

Then, follow similar steps mentioned in [the "Usage" section](#usage) above,
with the following differences for a CPU-only environment:
1. (Necessary) Add option `--device cpu` to indicate that only CPU is used when running the code.
2. (Optional) You may consider setting a larger timeout for CPU which is significantly slower than GPU, by `--override_timeout TIMEOUT`, where `TIMEOUT` is the timeout threshold in seconds (the default value for GPU is 300s).
3. (Optional) You may consider reducing the batch size if the batch size value specified in the YAML configuration file (in the `batch_size` field under `solver`; if not specified in the configuration file, the default value would be 64) causes too much memory usage, by `--batch_size BATCH_SIZE`, where `BATCH_SIZE` is a smaller batch size value.
