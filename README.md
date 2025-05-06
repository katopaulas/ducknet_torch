# ducknet_torch

Basic adaptation of the Deep Understanding Convolutional Kernel DUCK-NET for binary classification.

## Paper
[![arXiv](https://img.shields.io/badge/arXiv-2407.16298-b31b1b.svg)](http://dx.doi.org/10.1038/s41598-023-36940-5)

In order to adapt the model for binary classification a prediction head has been added at the end of the model combined with a sigmoid activation.

# Usage

!git clone ##repository

from .model import DuckNet

instantiate the model and hf!