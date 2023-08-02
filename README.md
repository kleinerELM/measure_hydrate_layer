# Measure hydrate layer

A python script to measure the thickness of the C-S-H layer around alite (C<sub>3</sub>S) or belite (C<sub>2</sub>S) particles.
This script is part of the paper _"A python tool to determine the thickness of the hydrate layer around clinker grains using SEM-BSE images."_ published for the 19th Euroseminar on Microscopy Applied to Building Materials 2024.

The dataset processed in this paper is also available and can be processed with this script.

## datasets

The datasets used are available at Zenodo
 - C<sub>3</sub>S (1,7,14, 28, 84, and 365 days)
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8118931.svg)](https://doi.org/10.5281/zenodo.8118931)
 - C<sub>2</sub>S (1,7,14, 28, 84, and 365 days)
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8119086.svg)](https://doi.org/10.5281/zenodo.8119086)


## required packages
This project was written with Python 3.10 and Jupyter notebooks in mind. Since Python 3.11 provides some significant performance improvements, it is recommended to use this version or newer.

The project depends on another repository. Just clone it to the same parent folder as this project.

```
git clone https://github.com/kleinerELM/tiff_scaling.git
```
The folder structure should look somewhat like this:

```
.
├── tiff_scaling
│   └── ...
└── measure_hydrate_layer
    └── ...
```

Additionally, some third-party packages are required.

###pip
You can install them either using pip:

```
pip install -r requirements-pip.txt
```
###(Ana)conda
or using anaconda. First switch to the desired enviornment (here `py311`):

```
conda activate py311
```

and then install the required packages:

```
conda install --yes --file requirements.txt
```