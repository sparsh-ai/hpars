# hash:sha256:e4fb39ebfe1cc7bb9180be7d2754fbdfd799d1f2f8a1f162144ff627bee37ed9
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.3.4 \
        numpy==1.19.2 \
        pandas==1.2.2 \
        scikit-learn==0.23.2 \
        seaborn==0.11.1 \
    && conda clean -ya
