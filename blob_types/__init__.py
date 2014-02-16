"""
The [blob_types](https://github.com/abbgrade/blob_types) package is structured into three modules.

- [Types](./types.html) contains abstract classes which help to build serializable data structures.
- [Interface](./interface.html) contains classes which generate c structs and functions for access to blob_types based data structures.
- [Utils](./utils.html) contains helper functions.
"""

__author__ = 'Steffen Kampmann'
__email__ = 'steffen.kampmann@gmail.com'
__version__ = '0.14.1'
__license__ = 'GPL 2'

import utils
import types

from utils import flat_struct, get_blob_index
from types import Blob, BlobArray, BlobLinkedListHost, BlobLinkedList, BlobEnum, \
    process_dtype_params, validate_dtype_params
from interface import BlobLib, FileLib, Lib as Lib
