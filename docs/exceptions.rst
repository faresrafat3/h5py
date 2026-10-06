.. currentmodule:: h5py
.. _exceptions:


Exceptions
==========

h5py uses builtin Python exceptions.  The table below documents which
exceptions each high-level API raises, so callers can catch precisely.

File
----

``h5py.File(name, mode='r')``

===================  ==================================================
Exception           When
===================  ==================================================
``FileNotFoundError``  mode is ``'r'`` or ``'r+'`` and the file does
                       not exist.
``FileExistsError``    mode is ``'x'`` and the file already exists.
``ValueError``         mode is not one of ``'r'``, ``'r+'``, ``'w'``,
                       ``'w-'``, ``'x'``, ``'a'``; or the driver name is
                       not registered.
``TypeError``          the file image does not support the buffer.
``OSError``            general I/O failure from the OS or HDF5.
===================  ==================================================

``f.close()`` raises ``ValueError`` if called on an already-closed file.

Group
-----

``g.create_group(name)``

===================  ==================================================
Exception           When
===================  ==================================================
``ValueError``         the identifier does not refer to a group.
``TypeError``          name is not a string or bytes.
``TypeError``          an incompatible object already exists at name.
===================  ==================================================

``g.create_dataset(name, ...)``

===================  ==================================================
Exception           When
===================  ==================================================
``ValueError``         name already exists in the group.
``TypeError``          dtype is not a type NumPy can interpret.
``TypeError``          shape is not a valid shape tuple.
===================  ==================================================

``g[name]`` (access)

===================  ==================================================
Exception           When
===================  ==================================================
``KeyError``           name does not exist in the group.
``ValueError``         the identifier does not refer to a group.
===================  ==================================================

Dataset
-------

``dset[...]`` (read/write)

===================  ==================================================
Exception           When
===================  ==================================================
``ValueError``         an invalid slice, an unknown file mode, or an
                       incompatible shape.
``TypeError``          an operation is not supported for the given
                       object (e.g. ``read_direct`` on an empty dataset).
``IndexError``         an index is out of range for a dataset or shape
                       tuple.
``RuntimeError``       the underlying HDF5 library reports a failure
                       that does not map to a more specific exception.
``NotImplementedError``  a feature is not available in the version of
                       the HDF5 library h5py was built against.
``OSError``            general I/O failure.
``OverflowError``      a value does not fit in the target datatype.
===================  ==================================================

``dset.astype(dtype)``

===================  ==================================================
Exception           When
===================  ==================================================
``TypeError``          dtype is not a type NumPy can interpret.
===================  ==================================================

``dset.read_direct(arr)``

===================  ==================================================
Exception           When
===================  ==================================================
``TypeError``          the dataset does not hold an HDF5 string
                       datatype; or the target array cannot match the
                       dataset shape.
===================  ==================================================

Attributes
----------

``attrs.create(name, data)``

===================  ==================================================
Exception           When
===================  ==================================================
``TypeError``          name is not a string or bytes.
``ValueError``         the attribute already exists with a different
                       shape or dtype.
``OSError``            the value cannot be stored in the existing
                       attribute's type.
===================  ==================================================

``attrs[name]`` (access)

===================  ==================================================
Exception           When
===================  ==================================================
``KeyError``           name does not exist in the attribute manager.
===================  ==================================================

Low-level
---------

The low-level interface (``h5py.h5*``) raises the same builtin
exceptions.  By default h5py prints the HDF5 error stack to stderr
whenever a low-level call fails.  This can be controlled with the
private ``h5py._errors`` module::

    import h5py._errors

    h5py._errors.silence_errors()    # stop printing HDF5 error stacks
    h5py._errors.unsilence_errors()  # resume printing them

These functions are not part of the public API and may change between
releases.

Catching errors
---------------

Because h5py uses builtin exceptions, a catch-all ``except Exception``
is rarely necessary.  Prefer the specific class::

    import h5py

    try:
        with h5py.File("data.h5", "r") as f:
            dset = f["measurements"]
    except FileNotFoundError:
        ...  # the file is not there
    except KeyError:
        ...  # the file is there, but has no "measurements" dataset
