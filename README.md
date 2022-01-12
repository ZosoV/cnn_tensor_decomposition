# Covariance Tensor for Convolutional Neural Network


## Installation

Install opencv and eigen library

```
apt install libopencv-dev
apt install libeigen3-dev
```

Compile a new `libtensorfilters.so` if it is needed. Go to the folder `filter_extract_lib`

```
make clean
make
```

Create and environment and install the needed packages using conda environment.

```bash
conda env create -f environment.yml
```

Or if your a using a pip environment use the requirements.txt
```bash
pip install -r requirements.txt
```

