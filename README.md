# ADVECTOR
One-stop-shop for all your plastic advection needs.  The goal here is to create a 3d plastic advection model which supports all the needs of the various TOC teams.  Research, for global or regional studies, Eng, for doing forecasting/analysis of the GPGP capture systems, [insert more use cases here].  Let's get rolling, without reinventing the wheel.

## Setup
You can set up your conda environment with `conda env create -f environment.yml`. Get miniconda [here](https://docs.conda.io/en/latest/miniconda.html).

Depending on your use-case, you may need to create a .env file with the following keys:
```
copernicus_uname=<username>  # for downloading cmems data
copernicus_pwd=<password>
```
