# Run GenBaB on all the experimented benchmarks

cd alpha-beta-CROWN/complete_verifier

# CIFAR-10 feedforward models
python abcrown.py --config ../../benchmarks/cifar/sigmoid_4fc_100/config.yaml
python abcrown.py --config ../../benchmarks/cifar/sigmoid_4fc_500/config.yaml
python abcrown.py --config ../../benchmarks/cifar/sigmoid_6fc_100/config.yaml
python abcrown.py --config ../../benchmarks/cifar/sigmoid_6fc_200/config.yaml
python abcrown.py --config ../../benchmarks/cifar/tanh_4fc_100/config.yaml
python abcrown.py --config ../../benchmarks/cifar/tanh_6fc_100/config.yaml
python abcrown.py --config ../../benchmarks/cifar/sin_4fc_100/config.yaml
python abcrown.py --config ../../benchmarks/cifar/sin_4fc_200/config.yaml
python abcrown.py --config ../../benchmarks/cifar/sin_4fc_500/config.yaml
python abcrown.py --config ../../benchmarks/cifar/gelu_4fc_100/config.yaml
python abcrown.py --config ../../benchmarks/cifar/gelu_4fc_200/config.yaml
python abcrown.py --config ../../benchmarks/cifar/gelu_4fc_500/config.yaml

# MNIST feedforward models from ERAN
python abcrown.py --config ../../benchmarks/eran/sigmoid_6_100/config.yaml
python abcrown.py --config ../../benchmarks/eran/sigmoid_6_200/config.yaml
python abcrown.py --config ../../benchmarks/eran/sigmoid_9_100/config.yaml
python abcrown.py --config ../../benchmarks/eran/sigmoid_conv_small/config.yaml
python abcrown.py --config ../../benchmarks/eran/tanh_6_100/config.yaml
python abcrown.py --config ../../benchmarks/eran/tanh_6_200/config.yaml
python abcrown.py --config ../../benchmarks/eran/tanh_9_100/config.yaml
python abcrown.py --config ../../benchmarks/eran/tanh_9_100/config.yaml

# LSTM
python abcrown.py --config ../../benchmarks/cifar/lstm_16_32/config.yaml
python abcrown.py --config ../../benchmarks/cifar/lstm_16_64/config.yaml
python abcrown.py --config ../../benchmarks/prover/mnist_lstm_7_32_1/config.yaml

# ViT
python abcrown.py --config ../../benchmarks/cifar/vit_1_3/config.yaml
python abcrown.py --config ../../benchmarks/cifar/vit_1_6/config.yaml
python abcrown.py --config ../../benchmarks/cifar/vit_2_3/config.yaml
python abcrown.py --config ../../benchmarks/cifar/vit_2_6/config.yaml

# ML4ACOPF
python abcrown.py --config ../../benchmarks/ml4acopf/config.yaml
