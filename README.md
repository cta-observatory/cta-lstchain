# cta-lstchain

Repository for the high level analysis of the LST.    
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.

master branch status: [![Build Status](https://travis-ci.org/cta-observatory/cta-lstchain.svg?branch=master)](https://travis-ci.org/cta-observatory/cta-lstchain)

  
Note that notebooks are currently not tested and not guaranteed to be up-to-date.   
In doubt, refer to tested code and scripts: basic functions of lstchain (reduction steps R0-->DL1 and DL1-->DL2) 
are unit tested and should be working as long as the build status is passing.

## Install

- You will need to install [anaconda](https://www.anaconda.com/distribution/#download-section) first. 

- Create and activate the conda environment:
```
git clone https://github.com/cta-observatory/cta-lstchain.git
cd cta-lstchain
conda env create -f environment.yml
source activate cta
```

- Install lstchain:

```
pip install -e .
```


## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)    
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Use [GitHub Issues](https://github.com/cta-observatory/cta-lstchain/issues).


