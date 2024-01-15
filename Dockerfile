FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /root

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir

ENV PYTHONPATH "${PYTHONPATH}:/root/"
ENV PYTHONPYCACHEPREFIX=/tmp/cpython/
