FROM ubuntu:latest
EXPOSE 8080
ARG DEBIAN_FRONTEND=noninteractive

# Move to home directory.
WORKDIR /home/

# Install git and wget dependencies.
RUN apt-get update
RUN apt-get install -y sudo git wget

# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
# Warning! Anyone who gets their hands on this image will be able
# to retrieve this private key file from the corresponding image layer
ADD id_rsa /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Add bitbuckets key
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone repository
RUN git clone git@github.com:AITrace/AITrace_prvt.git

# Move to app base dir.
WORKDIR /home/AITrace_prvt/ML

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

RUN protoc /home/AITrace_prvt/ML/AITrace/proto/aitrace.proto  --proto_path="/home/AITrace_prvt/ML/AITrace" --python_out="/home/AITrace_prvt/ML/AITrace"
RUN protoc /home/AITrace_prvt/ML/AITrace/proto/dataset.proto  --proto_path="/home/AITrace_prvt/ML/AITrace" --python_out="/home/AITrace_prvt/ML/AITrace"
RUN protoc /home/AITrace_prvt/ML/AITrace/proto/model.proto  --proto_path="/home/AITrace_prvt/ML/AITrace" --python_out="/home/AITrace_prvt/ML/AITrace"
RUN protoc /home/AITrace_prvt/ML/AITrace/proto/sampler.proto  --proto_path="/home/AITrace_prvt/ML/AITrace" --python_out="/home/AITrace_prvt/ML/AITrace"
RUN protoc /home/AITrace_prvt/ML/AITrace/proto/internal.proto  --proto_path="/home/AITrace_prvt/ML/AITrace" --python_out="/home/AITrace_prvt/ML/AITrace"

# Setup python environment
RUN mkdir -p /home/AITrace_prvt/ML/env
RUN python3.8 -m venv /home/AITrace_prvt/ML/env
RUN curl https://bootstrap.pypa.io/get-pip.py -o ./get-pip.py
RUN /home/AITrace_prvt/ML/env/bin/python3.8 ./get-pip.py
RUN /home/AITrace_prvt/ML/env/bin/python3.8 -m pip install wheel setuptools==53.1.0
RUN /home/AITrace_prvt/ML/env/bin/python3.8 -m pip install --upgrade pip==21.2.4
RUN /home/AITrace_prvt/ML/env/bin/python3.8 -m pip install --disable-pip-version-check --upgrade -r ./requirements.pip

RUN echo "##############################" >  "/home/AITrace_prvt/ML/aitrace"
RUN echo "# Auto generated bash binary #" >> "/home/AITrace_prvt/ML/aitrace"
RUN echo "# --------  AITrace ---------#" >> "/home/AITrace_prvt/ML/aitrace"
RUN echo "#       Not to be edited     #" >> "/home/AITrace_prvt/ML/aitrace"
RUN echo "##############################" >> "/home/AITrace_prvt/ML/aitrace"
RUN echo "eval PYTHONPATH=/home/AITrace_prvt/ML/ /home/AITrace_prvt/ML/env/bin/python3.8 /home/AITrace_prvt/ML/AITrace/aitrace.py \"\$@\"" >> "/home/AITrace_prvt/ML/aitrace"
RUN chmod 551 "/home/AITrace_prvt/ML/aitrace"

# Start application
CMD cd /home/AITrace_prvt/ML/ && git pull && /home/AITrace_prvt/ML/aitrace --workspace_dir final_workspace --config model_zoo/smoke_test.pbtxt --pt_cpu_only