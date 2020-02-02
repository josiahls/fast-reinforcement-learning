FROM nvidia/cuda:10.0-base-ubuntu18.04
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl vim ca-certificates python-qt4 libjpeg-dev \
        zip nano unzip libpng-dev strace python-opengl xvfb && \
        rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     apt-get update && apt-get upgrade -y --no-install-recommends

ENV PATH=$PATH:/opt/conda/bin/
ENV USER fastrl_user
# Create Enviroment
COPY environment.yaml /environment.yaml
RUN conda env create -f environment.yaml

# Cleanup
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get -y autoremove

EXPOSE 8888
ENV CONDA_DEFAULT_ENV fastrl

CMD ["/bin/bash -c"]
