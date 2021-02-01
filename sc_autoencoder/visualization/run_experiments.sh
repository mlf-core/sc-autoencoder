#!/bin/sh
#===============================================================#
# Perform experiments with different reproducibility settings
#===============================================================#

#Mapping of Experiment IDs to Experiment Names


## ID: 1 --> gpu_non_deterministic
#- gpu deterministic settings disabled
for var in 1 2 3 4 5 6 7 8 9 10; do mlflow run sc-autoencoder/ -A gpus=all --experiment-name  gpu_non_deterministic; done;

## ID: 2 --> gpu_random
#- gpu deterministic settings disabled
#- setting different seeds every time:
for var in 1 2 3 4 5 6 7 8 9 10; do mlflow run sc-autoencoder/ -A gpus=all -P tensorflow-seed=$var  -P general-seed=$var --experiment-name  gpu_random; done;

## ID: 3 --> cpu_random
#- setting different seeds every time:
for var in 1 2 3 4 5 6 7 8 9 10; do mlflow run sc-autoencoder/ -P tensorflow-seed=$var  -P general-seed=$var --experiment-name  cpu_random; done;

## ID: 4 --> gpu_deterministic
#- normal deterministic settings
for var in 1 2 3 4 5 6 7 8 9 10; do mlflow run sc-autoencoder/ -A gpus=all --experiment-name  gpu_non_deterministic; done;
