# GenBaB: Neural Network Verification with Branch-and-Bound for General Nonlinearities

GenBaB is a framework for neural network verification with branch-and-bound for supporting *general nonlinearties*.
GenBaB is generally formulated so that the branch-and-bound can be applied to verification on general computational graphs with general nonlinearities.
GenBaB also leverages the new flexibility from general nonlinearities beyond piecewise linear ReLU to make smarter decisions for the branching, with an improved branching heuristic named BBPS for choosing neurons to branch, and a mechanism for optimizing branching points on nonlinear functions being branched.
GenBaB has been demonstrated on a wide range of NNs, including NNs with activation functions such as Sigmoid, Tanh, Sine and GeLU, as well as NNs involving multi-dimensional nonlinear operations such as LSTMs and Vision Transformers.
GenBaB has also enabled new applications beyond simple NNs, such as [AC Optimal Power Flow (ACOPF)](https://github.com/AI4OPT/ml4acopf_benchmark).

The paper of GenBaB has been accepted by the 31st International Conference on Tools and Algorithms for the Construction and Analysis of Systems [(TACAS 2025)](https://etaps.org/2025/conferences/tacas/):

Zhouxing Shi\*, Qirui Jin\*, Zico Kolter, Suman Jana, Cho-Jui Hsieh, Huan Zhang.
[**Neural Network Verification with Branch-and-Bound for General Nonlinearities**](https://arxiv.org/abs/2405.21063). *To appear in TACAS 2025.* (*Equal contribution)

```bibtex
@inproceedings{shi2025genbab,
  title={Neural Network Verification with Branch-and-Bound for General Nonlinearities},
  author={Shi, Zhouxing and Jin, Qirui and Kolter, Zico and Jana, Suman and Hsieh, Cho-Jui and Zhang, Huan},
  booktitle={International Conference on Tools and Algorithms for the Construction and Analysis of Systems},
  year={2025}
}
```

## Dependencies

### Obtaining α,β-CROWN with GenBaB integrated

The GenBaB algorithm is implemented into our comprehensive [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) toolbox (our paper considered α,β-CROWN without GenBaB as a baseline, but GenBaB is integrated into newer α,β-CROWN).

A copy of the updated α,β-CROWN with GenBaB has been included in this repository.
At this point, the main repository of α,β-CROWN is yet to be updated with our latest GenBaB, and thus please use the version included in this repository to run GenBaB for now. However, we still recommend checking the [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)  repository to learn more about α,β-CROWN with GenBaB.
Note that in our paper, "α,β-CROWN" refers

### Obtaining benchmarks

Benchmarks used in the GenBaB paper are hosted at a [HuggingFace repository](zhouxingshi/GenBaB_benchmarks). Download them to a `benchmarks` folder by:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/datasets/zhouxingshi/GenBaB benchmarks
```

### Setting up the environment

Python 3.10+ and PyTorch 2.0+ compatible with CUDA are required. GenBaB has been tested with Python 3.10 and PyTorch 2.0. We recommend using [miniconda](https://docs.anaconda.com/miniconda/) to setup a clean Python environment and [install PyTorch 2.0.0](https://pytorch.org/get-started/previous-versions/#linux-and-windows-24).

After setting up Python and PyTorch, install other dependencies by:
```bash
cd alpha-beta-CROWN
pip install -e .
cd complete_verifier
pip install -r requirements.txt
```

## Usage

Run code from the `alpha-beta-CROWN/complete_verifier` directory.
The basic usage of running GenBaB on a model is by running `abcrown.py`
with a YAML configuration file.
```bash
python abcrown.py --config CONFIG_FILE
```

For the [benchmarks](https://huggingface.co/datasets/zhouxingshi/GenBaB/) we used,
a configuration file has already been included in each benchmark folder.
For example, to run GenBaB on the Sin 4x100 model:
```bash
python abcrown.py --config ../../benchmarks/cifar/sin_4fc_100/config.yaml
```

A list of commands to run GenBaB on all the experimented benchmarks is at [`run.sh`](./run.sh).

Before running commands in `run.sh`, it is recommended to run [`warmup.sh`](./warmup.sh) first. The warmup script will run each kind of model on a single instance with a short timeout to build the lookup table of pre-optimized branching points. Since this lookup table can be shared by all the instances with existing model architectures, the warmup step can separate the time cost of building the lookup table from the main experiments. Otherwise, the cost of pre-optimizing branching points may be counted toward the first instance of each new model architecture.

### Variants

Options to run variants of GenBaB or the baseline without branch-and-bound:
* `--complete_verifier skip`: Disable branch-and-bound.
* `--nonlinear_split_method babsr-like`: Use a BaBSR-like branching heuristic instead of BBPS proposed in the paper.
* `--branching_point_method uniform`: Disable optimized branching points.
* `--nonlinear_split_relu_only`: For models with a mix of ReLU and other nonlinearities, only consider branching ReLU neurons, not other nonlinearities.

## Running GenBaB on new models

The design of GenBaB is intended to be general for models containing various nonlinearities.
To run GenBaB on new models, it is recommended to prepare the model and specifications for verification following the general [VNN-COMP](https://github.com/verivital/vnncomp2024/issues/2#issue-2221794616) format.
Specifically, in a folder, models can be provided as [ONNX](https://onnx.ai/) files, and specifications should be provided using the [VNN-LIB](https://www.vnnlib.org/) format.
There should also be a CSV file `instances.csv` listing the instances, where each row in the CSV file contains the path to the ONNX file, VNN-LIB file, and the timeout (in seconds) for each instance.
See the example of [`ml4acopf`](https://huggingface.co/datasets/zhouxingshi/GenBaB/tree/main/ml4acopf), as well as all the benchmarks used in [VNN-COMP 2024](https://github.com/ChristopherBrix/vnncomp2024_benchmarks).

A configuration file is needed, and by default, you may use [`default_config.yaml`](./default_config.yaml). Then, follow the [usage](#usage) to run GenBaB.
