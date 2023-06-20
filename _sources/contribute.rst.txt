.. _contribute:

How to Contribute
=================

Follow the steps below to contribute to the project.

Basics git command for implementing a new feature:
 * Cloning the repository:
   ``git clone https://github.com/cta-observatory/cta-lstchain.git``
 * Checking out a branch:
   ``git switch -c mybranch``
 * Download the private test data using ``./download_test_data.sh``,
   ask one of the maintainers if you do not know the credentials.
 * Make the changes in ``<file>``
 * Run the unit tests
 * Committing your changes:
   ``git add <file>``, ``git commit``
 * Submitting changes:
   ``git push -u origin mybranch``

Unit tests
----------
Perform tests locally with ``pytest`` before committing the changed file.

Running just ``pytest`` will not run the tests requiring the private test data,
but as these are important, as these include for example all tests on observed LST data,
run ``pytest -m 'private_data or not private_data'``, to exectute all tests.

Build the documentation locally
-------------------------------
Install the required dependencies for building the documentation: ``pip install -e ".[docs]"``

Compile the documentation from ``docs``: ``make clean html``.

To look at the documentation in your browser, this command is handy:

.. code::

    python -m http.server -d _build/html

Then visit the printed url.

.. note::
    Automatic documentation deployment is actually doing:
    ``make html SPHINXOPTS="-W --keep-going -n --color -j auto"``


Contribute to the documentation
-------------------------------

The documentation might fall behind the code or analysis workflow, 
so if you find something that is not documented or not clear, please contribute to the documentation.    
Simply modify the corresponding `.rst` files in `docs` and open a pull-request with your changes.    

Please do, someone will probably come after you and face the same issues you did.