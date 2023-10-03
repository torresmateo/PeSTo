FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app

COPY . . 

RUN apt update && apt install -y build-essential

RUN conda create -n pestocus python=3.9 

SHELL ["conda", "run", "-n", "pestocus", "/bin/bash", "-c"]

RUN conda install pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia \
    && pip install gemmi \ 
    && conda install -c conda-forge tensorboard \ 
    && conda install numpy scipy pandas matplotlib scikit-learn \ 
    && conda install -c conda-forge tqdm \ 
    && conda install h5py


