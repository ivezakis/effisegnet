# EffiSegNet Implementation Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10601024.svg)](https://doi.org/10.5281/zenodo.10601024)

## Requirements

The environment.yml file contains the list of packages required to run the code. To create the environment, run the following command:

```bash
conda env create -f environment.yml
```

## Dataset

The dataset used for training, validation, and testing is the Kvasir-SEG dataset. The dataset can be downloaded from [here](https://datasets.simula.no/kvasir-seg/). 
It contains 1000 images of the gastrointestinal tract and their corresponding ground truth polyp segmentation masks.
For easy reproduction of our results, it is divided into 800 training, 100 validation, and 100 testing images.
