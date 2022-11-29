FROM ubuntu:latest
EXPOSE 8080
ENV TZ=Europe/Athens
ARG DEBIAN_FRONTEND=noninteractive

# Setup TZ.
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Move to home directory.
WORKDIR /home/

# Install git and wget dependencies.
RUN apt-get update
RUN apt-get install -y sudo git wget

# Clone repository
RUN git clone https://github.com/fivosts/AITrace.git

# Move to app base dir.
WORKDIR /home/AITrace

# Install all dependencies
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN sudo apt-get install -y \
  vim \
  build-essential \
  ca-certificates \
  curl \
  wget \
  protobuf-compiler \
  flex \
  nasm \
  tar \
  m4 \
  pkg-config \
  python3.8 \
  python3.8-dev \
  python3.8-venv \
  python3-distutils \
  python3-numpy \
  pv \
  zlib1g-dev \
  lib32ncurses5-dev \
  libexempi-dev \
  libmpc-dev \
  libboost-all-dev \
  libmysqlclient-dev \
  libsqlite3-dev \
  zip \
  unzip

RUN protoc /home/AITrace/AITrace/proto/aitrace.proto  --proto_path="/home/AITrace/AITrace" --python_out="/home/AITrace/AITrace"
RUN protoc /home/AITrace/AITrace/proto/dataset.proto  --proto_path="/home/AITrace/AITrace" --python_out="/home/AITrace/AITrace"
RUN protoc /home/AITrace/AITrace/proto/model.proto    --proto_path="/home/AITrace/AITrace" --python_out="/home/AITrace/AITrace"
RUN protoc /home/AITrace/AITrace/proto/sampler.proto  --proto_path="/home/AITrace/AITrace" --python_out="/home/AITrace/AITrace"
RUN protoc /home/AITrace/AITrace/proto/internal.proto  --proto_path="/home/AITrace/AITrace" --python_out="/home/AITrace/AITrace"

# Setup python environment
RUN mkdir -p /home/AITrace/env
RUN python3.8 -m venv /home/AITrace/env
RUN curl https://bootstrap.pypa.io/get-pip.py -o ./get-pip.py
RUN /home/AITrace/env/bin/python3.8 ./get-pip.py
RUN /home/AITrace/env/bin/python3.8 -m pip install wheel setuptools==53.1.0
RUN /home/AITrace/env/bin/python3.8 -m pip install --upgrade pip==21.2.4
RUN /home/AITrace/env/bin/python3.8 -m pip install --disable-pip-version-check --upgrade -r ./requirements.pip

RUN echo "##############################" >  "/home/AITrace/aitrace"
RUN echo "# Auto generated bash binary #" >> "/home/AITrace/aitrace"
RUN echo "# --------  AITrace ---------#" >> "/home/AITrace/aitrace"
RUN echo "#       Not to be edited     #" >> "/home/AITrace/aitrace"
RUN echo "##############################" >> "/home/AITrace/aitrace"
RUN echo "eval PYTHONPATH=/home/AITrace/ /home/AITrace/env/bin/python3.8 /home/AITrace/AITrace/aitrace.py \"\$@\"" >> "/home/AITrace/aitrace"
RUN chmod 551 "/home/AITrace/aitrace"

# Start application
CMD cd /home/AITrace/ && git pull && /home/AITrace/aitrace --workspace_dir native_workspace --config model_zoo/smoke_test_native.pbtxt --pt_cpu_only --checkpoint_step 19263 --only_sample
