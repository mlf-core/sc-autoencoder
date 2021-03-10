==============
sc-autoencoder
==============

.. image:: https://github.com/mlf-core/sc-autoencoder/workflows/Train%20sc-autoencoder%20using%20CPU/badge.svg
        :target: https://github.com/mlf-core/sc-autoencoder/actions?query=workflow%3A%22Train+sc-autoencoder+using+CPU%22
        :alt: Github Workflow CPU Training sc-autoencoder Status

.. image:: https://github.com/mlf-core/sc-autoencoder/workflows/Publish%20Container%20to%20Docker%20Packages/badge.svg
        :target: https://github.com/mlf-core/sc-autoencoder/actions?query=workflow%3A%22Publish+Container+to+Docker+Packages%22
        :alt: Publish Container to Docker Packages

.. image:: https://github.com/mlf-core/sc-autoencoder/workflows/mlf-core%20linting/badge.svg
        :target: https://github.com/mlf-core/sc-autoencoder/actions?query=workflow%3A%22mlf-core+lint%22
        :alt: mlf-core lint


.. image:: https://readthedocs.org/projects/sc-autoencoder/badge/?version=latest
        :target: https://sc-autoencoder.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

An autoencoder for single cell data.

* Free software: MIT
* Documentation: https://sc-autoencoder.readthedocs.io.

This project uses an autoencoder model to learn latent features from single-cell RNA-seq (scRNA-seq) data. Autoencoder models
and similar architectures are frequently used for scRNA-seq data. For instance, Eraslan et al. used an `autoencoder <https://www.nature.com/articles/s41467-018-07931-2>`_ for denoising
of single cell data. In another study, Lotfollahi et al use a `variational autoencoder <https://www.nature.com/articles/s41592-019-0494-8>`_ to predict perturbation responses.
Here, we have implemented a very simple autoencoder to demonstrate how non-deterministic operations can lead to significant differences in latent space embeddings which affect
downstream analysis and hinder reproducibility.

Architecture
------------
The model used in this project follows a standard encoder-encoding-decoder autoencoder architecture.
We use layer sizes of 256, 128 and 64 for the encoder and decoder (in reverse) layers, and a encoding size of 32.


.. image:: docs/images/autoencoder_architecture.png
        :alt: Autoencoder architecture
        :scale: 10


Credits
-------

This package was created with `mlf-core`_ using Cookiecutter_.

.. _mlf-core: https://mlf-core.readthedocs.io/en/latest/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
