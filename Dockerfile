FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CONDA_DIR=/opt/conda
ENV CONDA_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH}
ENV PLL_INC_DIR=/usr/local/include
ENV PLL_LIB_DIR=/usr/local/lib
ENV EPA_NG_ENV=base
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace/MLIPPER

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    bison \
    flex \
    git \
    libblas-dev \
    liblapack-dev \
    libtbb-dev \
    libtool \
    pkg-config \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y -n base -c bioconda epa-ng \
    && conda clean -afy

RUN git clone --depth 1 https://github.com/xflouris/libpll.git /tmp/libpll \
    && cd /tmp/libpll \
    && if [[ -x ./autogen.sh ]]; then ./autogen.sh; else autoreconf -fi; fi \
    && ./configure --prefix=/usr/local \
    && make -j1 \
    && make install \
    && rm -rf /tmp/libpll \
    && ldconfig

COPY . /workspace/MLIPPER

RUN make float

CMD ["/bin/bash"]
