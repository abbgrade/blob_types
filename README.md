blob_types
==========

A Python lib, which parses a JSON structure into full featured Python objects.
It enables to pass them to a OpenCL devices and to generate required C structures, initialization, serialization and deserialization functions.

Contribute
----------

[GitHub Project](https://github.com/abbgrade/blob_types)

Install
-------

	python setup.py install

Documentation
-------------

The [doc](http://abbgrade.github.io/blob_types/__init__.html) is based on [pycco](https://github.com/fitzgen/pycco) and hosted on github pages.

The generated HTML is stored in the *gh-pages* branch.

	git clone https://github.com/abbgrade/blob_types/tree/gh-pages# blob_types_doc

It could be generated with the *pycco* command.

	pycco blob_types/blob_types/*.py -d blob_types_doc

TODO
----

 - reduce usage of issubclass (its to expensive)
 - reduce usage (and / or) optimize utils.flat_struct
 - reduce convertion of dtype_params to dtype
 - reduce change get_dtype_param_keys to static variable

