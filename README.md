# cta-lstchain [![Build Status](https://github.com/cta-observatory/cta-lstchain/workflows/CI/badge.svg?branch=main)](https://github.com/cta-observatory/cta-lstchain/actions?query=workflow%3ACI+branch%3Amain) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6344673.svg)](https://doi.org/10.5281/zenodo.6344673) [![PyPI version](https://badge.fury.io/py/lstchain.svg)](https://badge.fury.io/py/lstchain) [![Conda version](https://anaconda.org/conda-forge/lstchain/badges/version.svg)](https://anaconda.org/conda-forge/lstchain)

Repository for the low-level analysis of the LST up to DL3 level.
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction. Higher-level analysis starting from DL3 can be performed with [Gammapy](https://gammapy.org/).

- **Source code:** https://github.com/cta-observatory/cta-lstchain
- **Documentation:** https://cta-observatory.github.io/cta-lstchain/

Note that notebooks are currently not tested and not guaranteed to be up-to-date.   
In doubt, refer to tested code and scripts: basic functions of lstchain (reduction steps R0-->DL1, DL1-->DL2 and DL2-->DL3) 
are unit-tested and should be working as long as the build status is passing.

## Install

You will need to install [micromamba/mamba](https://mamba.readthedocs.io/en/latest/installation.html) (recommended), [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com/distribution/#download-section) first.


### As user

You can create an environment and install `lstchain` from conda-forge as:
```
mamba create -c conda-forge -n lstchain-v0.10.7 python=3.11 lstchain=0.10.7
```

Alternatively, you can also install `lstchain` from PyPi with `pip`:
```
LSTCHAIN_VER=0.10.7  (or the version you want to install - usually the latest release)
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

**Note**: To prevent packages you installed with `pip install --user` from taking precedence over the conda environment, run:
```
conda env config vars set PYTHONNOUSERSITE=1 -n <environment_name>
```

To update the environment (e.g. when dependencies got updated), use:
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
Ask one of the project maintainers for the credentials. If 
you are a member of the LST collaboration you can also obtain them here:

https://ctaoobservatory.sharepoint.com/:i:/r/sites/ctan-onsite-it/Shared%20Documents/General/information_2.jpg?csf=1&web=1&e=suUkV6

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

All contributions are welcomed.

Guidelines are the same as [ctapipe's ones](https://ctapipe.readthedocs.io/en/latest/developer-guide/index.html). See [here](https://ctapipe.readthedocs.io/en/latest/developer-guide/pullrequests.html) for the general guidelines on how to make a pull request to contribute to the repository. Since the addition of the private data, the CI tests for Pull Requests from forks are not working, therefore we would like to ask to push your modified branches directly to the main cta-lstchain repo. If you do not have writing permissions in the repo, please contact one of the main developers. 


## Report issue / Ask a question

Use [GitHub Issues](https://github.com/cta-observatory/cta-lstchain/issues).

## Cite

If you use lstchain in a publication, please cite the exact version you used from Zenodo _Cite as_, see https://doi.org/10.5281/zenodo.6344673

Please also cite the following proceedings by adding the bibtex entry:

```
@inproceedings{lst_performance_icrc2021,
  author = {López-Coto, R. and Moralejo, A. and Artero, M. and Baquero, A. and Bernardos, M. and Contreras, J. L. and Di Pierro, F. and García, E. and Kerszberg, D. and López-Moya, M. and MasAguilar, A. and Morcuende, D. and Noethe, M. and Nozaki, S. and Ohtani, Y. and Priyadarshi, C. and Suda, Y. and Vuillaume, T. and others},
  usera = "{for the CTA LST Project}",
  title = "{Physics Performance of the Large Size Telescope prototype of the Cherenkov Telescope Array}",
  doi = "10.22323/1.395.0806",
  booktitle = "Proceedings, 37th International Cosmic Ray Conference",
  location = "Berlin, Germany",
  year = 2021,
  volume = "395",
  pages = "806"
}
```

and the macro to your main `.tex` file to correctly add the "CTA LST Project":
```
% we use the user a field as "dedication", i.e. for the CTA-LST Consortium
\renewbibmacro*{author}{%
  \iffieldundef{usera}{\printnames{author}}{\printnames{author} \printfield{usera}}%
}%
```
