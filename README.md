# Measure hydrate layer

A python script to measure the thickness of the C-S-H layer around alite (C<sub>3</sub>S) or belite (C<sub>2</sub>S) particles.
This script is part of the paper __"A python tool to determine the thickness of the hydrate layer around clinker grains using SEM-BSE images."__, (to be) published for the _19<sup>th</sup> Euroseminar on Microscopy Applied to Building Materials 2024_.
The preprint is available at

Open the Jupyter notebook `cement_hydrate_fringe_C3S 7d-example.ipynb` to see an exaple on how to process the datasets.
The notebook `final_evaluation_C3S.ipynb` includes some postprocessing/evaluation of the preprocessed datasets.

## datasets

The datasets processed using this script and mentioned in the corresponding paper are available at Zenodo:
 - C<sub>3</sub>S (1,7,14, 28, 84, and 365 days)
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8118931.svg)](https://doi.org/10.5281/zenodo.8118931)
 - C<sub>2</sub>S (14, 28, 84, and 365 days)
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8119086.svg)](https://doi.org/10.5281/zenodo.8119086)
 - C<sub>3</sub>S (14 days, reduced resolution, segmented)
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10648006.svg)](https://doi.org/10.5281/zenodo.10648006)

The datasets should be saved in the following folder structure:

```
[tiff_scaling]
├── [C3S]
│   ├── [1d]
│   │   └── C3S 1d.tif
│   ├── [7d]
│   │   ├── 7d_example.tif
│   │   ├── C3S 7d.tif
│   │   └── C3S 7d_2.tif
│   ├── [14d]
│   │   ├── [guided_segmentation]
│   │   │   ├── C3S.tif
│   │   │   ├── CH.tif
│   │   │   ├── CSH.tif
│   │   │   └── pores.tif
│   │   ├── ...
│   │   ├── C3S 14d.tif
│   │   └── C3S 14d_half-resolution.tif
│   └── ...
├── [C2S]
│   └── ...
└── *.*
```


## required packages
This project was written with Python 3.10 and Jupyter notebooks in mind. Since Python 3.11 provides some significant performance improvements, it is recommended to use this version or newer.

The project depends on another repository. Just clone it to the same parent folder as this project.

```
git clone https://github.com/kleinerELM/tiff_scaling.git
```
The folder structure for the scripts should look somewhat like this:

```
[.]
├── [tiff_scaling]
│   └── ...
└── [measure_hydrate_layer]
    └── ...
```

Additionally, some third-party packages are required.

__pip__
You can install them either using pip:

```
pip install -r requirements-pip.txt
```
__(Ana)conda__
or using Anaconda.
First switch to the desired environment (here `py311`):

```
conda activate py311
```

and then install the required packages:

```
conda install --yes --file requirements.txt
```