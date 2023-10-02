FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app

COPY . . 

RUN conda create -n pestocus python=3.9 

SHELL ["conda", "run", "-n", "pestocus", "/bin/bash", "-c"]

RUN conda install pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda install conda-forge::gemmi
RUN conda install conda-forge::tensorboard
RUN conda install numpy scipy pandas matplotlib scikit-learn conda-forge::tqdm
RUN conda install anaconda::h5py


