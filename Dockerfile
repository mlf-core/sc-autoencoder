FROM mlfcore/base:1.0.0

# Install the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
RUN echo "source activate sc_autoencoder" >> ~/.bashrc
ENV PATH /home/user/miniconda/envs/sc-autoencoder/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name sc-autoencoder > sc-autoencoder_environment.yml

# Currently required, since mlflow writes every file as root!
USER root
