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

# Configure conda channels (order matches environment.yml: rapidsai, pytorch, conda-forge, nvidia)
RUN conda config --set channel_priority flexible && \
    conda config --add channels nvidia && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels rapidsai

WORKDIR /app

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install core scientific + ML stack into base env
RUN conda install -y \
    python=3.11 \
    "numpy>=1.24,<3.0" \
    "scipy>=1.10,<2.0" \
    "pandas>=1.5,<3.0" \
    "polars>=1.0" \
    "pyarrow>=15.0" \
    "scikit-learn>=1.2,<1.8" \
    "xgboost>=1.7,<3.0" \
    "statsmodels>=0.14" \
    "joblib>=1.2" \
    "rdkit>=2023.03" \
    "datamol>=0.12" \
    mordredcommunity \
    "h5py>=3.0" \
    "hdf5plugin>=3.0" \
    "matplotlib-base>=3.5" \
    "rich>=10.0" \
    "pyyaml>=5.0" \
    "pytest>=7.0" \
    "ruff>=0.4" \
    "mypy>=1.0" \
    ipython \
    ipykernel \
    setuptools \
    -c conda-forge && \
    conda clean -afy

# Install GPU stack: PyTorch + CUDA, gpytorch, RAPIDS cuml, cupy, treelite
# pytorch-cuda is a metapackage pinning the CUDA runtime PyTorch was built against.
RUN conda install -y \
    "pytorch>=2.0,<3.0" \
    "pytorch-cuda>=12.4" \
    "pytorch-lightning>=2.0,<3.0" \
    "gpytorch>=1.11" \
    "treelite>=4.4,<5.0" \
    cupy \
    -c pytorch -c nvidia -c conda-forge && \
    conda clean -afy

# RAPIDS cuML (GPU-accelerated scikit-learn) installed separately
# from rapidsai channel to avoid solver conflicts with the pytorch stack above.
RUN conda install -y \
    "cuml>=25.04" \
    -c rapidsai -c conda-forge -c nvidia && \
    conda clean -afy

# Install pip-only packages (chemprop, fastprop, scikit-fingerprints, gauche, bitbirch)
RUN pip install --no-cache-dir \
    chemprop \
    fastprop \
    scikit-fingerprints \
    "gauche>=0.1.6" \
    "pytest-xdist>=3.5" \
    git+https://github.com/mqcomplab/bitbirch.git

# Copy project and install in editable mode
COPY . .
RUN pip install -e .

CMD ["/bin/bash"]