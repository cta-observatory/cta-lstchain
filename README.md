# cta-lstchain

Repository for the high level analysis of the LST.    
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.

master branch status: [![Build Status](https://travis-ci.org/cta-observatory/cta-lstchain.svg?branch=master)](https://travis-ci.org/cta-observatory/cta-lstchain)


### Important message to lstchain users (May 4th 2019):
*ctapipe* and *lstchain* are currently undergoing heavy and rapid changes.    
The core developer team is trying to stay up-to-date with the master version of *ctapipe* before reaching *ctapipe v0.7* release.
You might experience some issues if changes have been merged in *ctapipe* master before we could integrate these changes in *lstchain*. We are sorry for that. Do not hesitate to submit an issue or propose a patch through a pull request.

- The basic functions of lstchain (reduction steps R0-->DL1 and DL1-->DL2) are unit tested and should be working as long as the build status is passing.    
- However, the notebooks are not and might not be up-to-date before stable release. Do not rely on them for now.



## Install

> Old install procedure:
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
git clone https://github.com/cta-observatory/cta-lstchain.git
cd cta-lstchain
conda env create --name cta --file environment.yml
conda activate cta
pip install https://github.com/cta-observatory/ctapipe/archive/master.tar.gz
pip install https://github.com/cta-sst-1m/protozfitsreader/archive/v1.4.2.tar.gz
pip install https://github.com/cta-observatory/ctapipe_io_lst/archive/master.tar.gz
pip install -e .
```


## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Use GitHub Issues.


