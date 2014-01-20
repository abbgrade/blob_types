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

TODO
----

 - reduce usage of issubclass (its to expensive)
 - reduce usage (and / or) optimize utils.flat_struct
 - reduce convertion of dtype_params to dtype
 - reduce change get_dtype_param_keys to static variable

