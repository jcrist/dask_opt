FROM ubuntu:latest

RUN apt-get update -yqq  && apt-get install -yqq \
    wget \
    bzip2 \
#  git \
#  libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configure environment
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install miniconda and python 3
RUN wget -O miniconda.sh \
  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash miniconda.sh -b -p /work/miniconda \
  && rm miniconda.sh

ENV PATH="/work/bin:/work/miniconda/bin:$PATH"

# install long-term requirements
RUN conda update -y python conda && \
    conda install -y \
    pip \
    setuptools \
    jupyter \
    numpy \
    scipy \
    pandas \
    bokeh \
    dask \
    distributed \
    scikit-learn

RUN pip install traces
#    && conda clean -tipsy

# Install matplotlib and scikit-image without Qt
#RUN conda update -y python conda && \
#  conda install -y --no-deps \
#  matplotlib \
#  cycler \
#  freetype \
#  libpng \
#  pyparsing \
#  pytz \
#  python-dateutil \
#  scikit-image \
#  networkx \
#  pillow \
#  six \
#  && conda clean -tipsy
#
#RUN conda install -y \
#  pip \
#  setuptools \
#  notebook \
#  ipywidgets \
#  terminado \
#  psutil \
#  numpy \
#  scipy \
#  pandas \
#  bokeh \
#  scikit-learn \
#  statsmodels \
#  && conda clean -tipsy

# copy source to image and python setup.py develop
COPY . /work/

# do python setup.py develop
RUN cd /work && python setup.py develop

# download the dataset for 20Newsgroups ^^^ up here
# set scikit-learn data directory
#COPY ./examples/output_data/scikit_learn_data /data/



