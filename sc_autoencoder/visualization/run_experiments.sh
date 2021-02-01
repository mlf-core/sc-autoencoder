#!/bin/sh
#===============================================================#
# Perform experiments with different reproducibility settings
#===============================================================#

#Mapping of Experiment IDs to Experiment Names


## ID: 1 --> gpu_non_deterministic
#- gpu deterministic settings disabled
echo "Running gpu_non_deterministic"
for var in 1; do mlflow run ../../ -A gpus=all --experiment-name  gpu_non_deterministic -P deterministic='0'; done;

## ID: 2 --> gpu_random
#- gpu deterministic settings disabled
#- setting different seeds every time:
echo "Running gpu_random"
for var in 1; do mlflow run ../../ -A gpus=all -P tensorflow-seed=$var  -P general-seed=$var --experiment-name  gpu_random -P deterministic='0'; done;

## ID: 3 --> cpu_random
#- setting different seeds every time:
echo "Running cpu_random"
for var in 1; do mlflow run ../../ -P tensorflow-seed=$var  -P general-seed=$var --experiment-name  cpu_random; done;

## ID: 4 --> gpu_deterministic
#- normal deterministic settings
echo "Running gpu_deterministic"
for var in 1; do mlflow run ../../ -A gpus=all --experiment-name  gpu_deterministic -P deterministic='1'; done;
