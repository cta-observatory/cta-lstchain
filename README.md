# cta-lstchain [![Build Status](https://github.com/cta-observatory/cta-lstchain/workflows/CI/badge.svg?branch=master)](https://github.com/cta-observatory/cta-lstchain/actions?query=workflow%3ACI+branch%3Amaster)

Repository for the high level analysis of the LST.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.


Note that notebooks are currently not tested and not guaranteed to be up-to-date.   
In doubt, refer to tested code and scripts: basic functions of lstchain (reduction steps R0-->DL1 and DL1-->DL2) 
are unit tested and should be working as long as the build status is passing.

## Install

- You will need to install [anaconda](https://www.anaconda.com/distribution/#download-section) first. 


### As user

```
LSTCHAIN_VER=0.7.0
wget https://raw.githubusercontent.com/cta-observatory/cta-lstchain/v$LSTCHAIN_VER/environment.yml
conda env create -n lst -f environment.yml
conda activate lst
pip install lstchain==$LSTCHAIN_VER
rm environment.yml
```

### As developer

- Create and activate the conda environment:
```
git clone https://github.com/cta-observatory/cta-lstchain.git
cd cta-lstchain
conda env create -f environment.yml
conda activate lst-dev
```

To update the environment (e.g. when depenencies got updated), use:
```
conda env update -n lst-dev -f environment.yml
```

- Install lstchain in developer mode:

```
pip install -e .
```

To run some of the tests, some non-public test data files are needed.
These tests will not be run locally if the test data is not available,
but are always run in the CI.

To download the test files locally, run `./download_test_data.sh`.
It will ask for username and password and requires `wget` to be installed.
Ask one of the project maintainers for the credentials.

To run the tests that need those private data file, add `-m private_data`
to the pytest call, e.g.:

```
pytest -m private_data -v lstchain
```

To run all tests, run
```
pytest -m 'private_data or not private_data' -v lstchain
```

## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)    
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) for the general guidelines on how to make a pull request to contribute to the repository. Since the addition of the private data, the CI tests for Pull Requests from forks are not working, therefore we would like to ask to push your modified branches directly to the main cta-lstchain repo. If you do not have writing permissions in the repo, please contact one of the main developers. 


## Report issue / Ask a question

Use [GitHub Issues](https://github.com/cta-observatory/cta-lstchain/issues).
