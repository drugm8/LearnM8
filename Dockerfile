# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Prevent timezone prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configure conda
RUN conda config --set channel_priority flexible && \
    conda config --add channels pytorch && \
    conda config --add channels nvidia && \
    conda config --add channels conda-forge

# Set working directory
WORKDIR /app

# Create a new environment.yml file


COPY environment.yml .
# Create conda environment
RUN conda env update -n base -f environment.yml && \
    conda clean -afy

# Verify CUDA installation
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CMD ["/bin/bash"]