.. _contribute:

How to Contribute
=================

Follow the steps below to contribute to the project.

Git
---
Basics git command for implementing a new feature:
 * Cloning the repository:
   ``git clone https://github.com/cta-observatory/cta-lstchain.git``
 * Checking out a branch:
   ``git checkout -b mybranch``
 * Make the changes in ``<file>``
 * Committing your changes:
   ``git add <file>``, ``git commit -m 'commit message'``
 * Submitting changes:
   ``git push origin mybranch``

Pytest
------
Perform tests locally with ``pytest`` before committing the changed file.

Build the documentation locally
-------------------------------
Install the required dependencies for building the documentation: ``pip install -e .[docs]``

Compile the documentation from ``docs``: ``make clean html`` and check the result in ``_build/html/index.html``

.. note::
    Automatic documentation deployment is actually doing:
    ``make html SPHINXOPTS="-W --keep-going -n --color -j auto"``