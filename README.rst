=======
cta-lstchain |buildstatus| |codacy| |coverage| |pypi|
=======

.. |buildstatus| image:: https://travis-ci.org/cta-observatory/cta-lstchain.svg?branch=master
    :target: https://travis-ci.org/cta-observatory/cta-lstchain
    :alt: Travis Build Status

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/c28d5fdc326e43b2961015b199f02d90)](https://www.codacy.com/gh/cta-observatory/cta-lstchain?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/cta-lstchain&amp;utm_campaign=Badge_Grade
    :alt: Code Quality

.. |coverage| image:: https://codecov.io/gh/cta-observatory/cta-lstchain/branch/master/graph/badge.svg 
     :target: https://codecov.io/gh/cta-observatory/cta-lstchain
     :alt: Code Coverage

.. |pypi| image:: https://img.shields.io/pypi/v/lstchain.svg
    :target: https://pypi.python.org/pypi/cta-lstchain
    :alt: cta-lstchain's PyPI Status


Repository for the high level analysis of the LST.    
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe), adding custom code for mono reconstruction.

master branch status: 

  
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
conda activate lst-dev
```

To update the environment (e.g. when depenencies got updated), use:
```
conda env update -n lst-dev -f environment.yml
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


