Code snippets
=============

Some code snippets that can be useful.


Print lstchain version used to produce a file
---------------------------------------------

The lstchain version is stored in the file metadata. You can print it with the following command:

In python: 

.. code-block:: python

    import tables
    filename = 'dl1_file.h5'
    file = tables.open_file(filename)
    print(file.root._v_attrs['LSTCHAIN_VERSION'])

In bash:

.. code-block:: bash

    h5dump -a /LSTCHAIN_VERSION dl1_file.h5

