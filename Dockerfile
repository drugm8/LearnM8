# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configure conda channels
RUN conda config --set channel_priority flexible && \
    conda config --add channels pytorch && \
    conda config --add channels nvidia && \
    conda config --add channels conda-forge

WORKDIR /app

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install conda packages into base env
RUN conda install -y python=3.11 \
    numpy=2.4.2 \
    pandas=3.0.1 \
    matplotlib=3.10.8 \
    scipy=1.17.1 \
    seaborn=0.13.2 \
    h5py=3.15.1 \
    joblib=1.5.3 \
    tqdm=4.67.3 \
    pyyaml=6.0.3 \
    xgboost=3.2.0 \
    ipython \
    ipykernel=7.2.0 \
    setuptools=82.0.0 \
    -c conda-forge && \
    conda clean -afy

# Install pip packages
RUN pip install --no-cache-dir \
    chemprop==2.2.2 \
    datamol==0.12.5 \
    fastprop==1.2.2 \
    hdf5plugin==6.0.0 \
    lightning==2.6.1 \
    meeko==0.7.1 \
    polars==1.38.1 \
    pytest==9.0.2 \
    pytorch_lightning==2.6.1 \
    rich==14.3.3 \
    xgboost==3.2.0 \
    scikit-fingerprints

# Copy project and install in editable mode
COPY . .
RUN pip install -e .

CMD ["/bin/bash"]