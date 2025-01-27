FROM ubuntu:latest

RUN apt update
RUN apt install -y tmux vim git git-lfs wget curl libicu-dev psmisc htop libssl-dev
RUN apt install -y m4 build-essential autoconf libtool unzip

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -u -p ~/miniconda3
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda create -y --name GenBaB python=3.11

RUN git clone --recursive https://github.com/shizhouxing/GenBaB.git
RUN git lfs install
RUN git clone https://huggingface.co/datasets/zhouxingshi/GenBaB GenBaB/benchmarks

# Build with "--build-arg CPU=1" to install CPU-version
ARG CPU
RUN if [ -z "$CPU" ] ; then echo "Default PyTorch will be installed"; \
else echo "CPU-version PyTorch will be installed" \
&& ~/miniconda3/envs/GenBaB/bin/pip install \
torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
--index-url https://download.pytorch.org/whl/cpu ; \
fi

RUN cd GenBaB/alpha-beta-CROWN/auto_LiRPA \
    && ~/miniconda3/envs/GenBaB/bin/pip install -e .
RUN cd GenBaB/alpha-beta-CROWN/complete_verifier \
    && ~/miniconda3/envs/GenBaB/bin/pip install -r requirements.txt

CMD bash
