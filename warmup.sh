# Warmup: quickly run each kind of models on a single instance with a short timeout.
# This is for building the lookup table of pre-optimized branching points.

cd alpha-beta-CROWN/complete_verifier

WARMUP="--init_iteration 2 --select_instance 0 --override_timeout 15"

python abcrown.py --config ../../benchmarks/cifar/sigmoid_4fc_100/config.yaml $WARMUP
python abcrown.py --config ../../benchmarks/cifar/tanh_4fc_100/config.yaml $WARMUP
python abcrown.py --config ../../benchmarks/cifar/sin_4fc_100/config.yaml $WARMUP
python abcrown.py --config ../../benchmarks/cifar/gelu_4fc_100/config.yaml $WARMUP
python abcrown.py --config ../../benchmarks/cifar/lstm_16_32/config.yaml $WARMUP
python abcrown.py --config ../../benchmarks/cifar/vit_1_3/config.yaml $WARMUP
