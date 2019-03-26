# cta-lstchain

Repository for the high level analysis of the LST.    
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.

master branch status: [![Build Status](https://travis-ci.org/cta-observatory/cta-lstchain.svg?branch=master)](https://travis-ci.org/cta-observatory/cta-lstchain)

## Install

> If you are a user and don't already have ctapipe installed:
> ```
> conda env create -f environment.yml
> source activate cta
> ```
> This will create a conda environment called `cta` and install ctapipe with all dependencies.

> Then you can install the `lstchain` in this environment with:
> ```
> python setup.py install
> ```

Current `lstchain` build uses `ctapipe` master version.   
Here is how you should install:
```
conda env create --name cta --file environment.yml
conda activate cta
pip install https://github.com/cta-observatory/ctapipe/archive/$CTAPIPE_VERSION.tar.gz
pip install https://github.com/cta-sst-1m/protozfitsreader/archive/$PROTOZFITS_VERSION.tar.gz
pip install https://github.com/cta-observatory/ctapipe_io_lst/archive/$CTAPIPE_IO_LST_VERSION.tar.gz
python setup.py install
```


## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Use GitHub Issues.


