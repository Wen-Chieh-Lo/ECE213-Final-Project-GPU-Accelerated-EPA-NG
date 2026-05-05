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
ENV RAXML_NG_ENV=base
# Add CUDA and /usr/local shared libraries to the runtime search path.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

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
    && conda install -y -n base -c bioconda epa-ng raxml-ng aster \
    && conda run -n base epa-ng --help >/dev/null \
    && conda run -n base raxml-ng --help >/dev/null \
    && conda run -n base astral-pro3 --help >/dev/null \
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
