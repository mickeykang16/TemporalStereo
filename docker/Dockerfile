FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.8 python3-pip
RUN apt-get -y install git
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install cupy-cuda113

RUN apt-get -y install qt5-default python3-pyqt5
RUN pip install pyqt5 --verbose
RUN pip install debugpy
RUN apt-get -y install net-tools

RUN mkdir -p /home/jaeyoung/data
RUN mkdir -p /home/jaeyoung/ws
ADD ../requirements.txt /home/jaeyoung/requirements.txt
ADD ../.tmux.conf ~/.tmux.conf
RUN apt-get -y install tmux

WORKDIR /home/jaeyoung/ws/TemporalEventStereo
RUN pip install -r ../../requirements.txt
# CMD python3 -m debugpy --listen 0.0.0.0:5678 /home/jaeyoung/ws/TemporalStereo/projects/TemporalStereo/dist_train.py --config-file /home/jaeyoung/ws/TemporalStereo/projects/TemporalStereo/configs/kitti2015-pair.yaml
# ENV CUDA_VISIBLE_DEVICES 4,5,6,7
RUN apt-get install htop
RUN pip install wandb
RUN pip install pillow==9.0.1
RUN apt-get install -y xclip
RUN echo "alias tn='. run_tmux.sh'" >> ~/.bashrc
# RUN cd .
# $ git clone https://github.com/NVIDIA/apex
# $ cd apex
# $ git reset --hard 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
# $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 