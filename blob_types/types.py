import logging
import numpy

from utils import flat_struct, camel_case_to_underscore, underscore_to_camel_case, get_blob_index, diff_dtype, vector_fields, implode_float_n

def validate_dtype_params(function):

    def wrapper(cls, dtype_params=None, *args, **kwargs):

        if dtype_params is None and cls.is_plain():
            dtype_params = {}

        assert isinstance(dtype_params, dict), 'expect dtype_params as dict instead of %s' % type(dtype_params)

        expected_keys = cls.get_dtype_param_keys()

        # validate dtype_params
        for key in expected_keys:
            if key not in dtype_params.keys():
                raise TypeError('%s requires argument %s' % (cls, key))

        return function(cls, *args, dtype_params=dtype_params, **kwargs)

    return wrapper


def process_dtype_params(function):

    def wrapper(cls, dtype=None, dtype_params=None, *args, **kwargs):

        if dtype_params is None and cls.is_plain():
            dtype_params = {}

        if dtype is None and dtype_params is None:
            raise TypeError('%s requires argument dtype or dtype_params not %s, %s' % (cls, args, kwargs))

        if dtype is None:

            assert isinstance(dtype_params, dict)

            # validate dtype_params
            for key in cls.get_dtype_param_keys():

                if key not in dtype_params.keys():
                    raise TypeError('%s requires argument %s' % (cls, key))

            if function.__name__ == 'create_dtype':

                for key, value in dtype_params.items():
                    assert value > 0, 'invalid value of %s (%d)' % (key, value)

                return function(cls, *args, dtype_params=dtype_params, **kwargs)

            if (hasattr(cls, 'subtypes') and len(cls.subtypes) > 0) or issubclass(cls, BlobArray):
                dtype = cls.create_dtype(dtype_params=dtype_params)

            elif hasattr(cls, 'dtype'):
                dtype = cls.dtype

            else:
                raise NotImplementedError()

        assert function.__name__ != 'create_dtype', 'create_dtype requires dtype_params, maybe the parameters are now named'

        return function(cls, *args, dtype_params=dtype_params, dtype=dtype, **kwargs)

    return wrapper


class BlobValidationException(Exception):
    pass


class EnumException(Exception):
    pass


class Blob(object):
    """Encapsulates an binary space as python object.
    
    A Blob is an abstract class, whose objects encapsulate and numpy.ndarray with a defined type structure.
    The type structure is based on numpy.dtypes and nested structures. 
    It manages the access to variables which are stored in the space of binary memory (blob).
    It generates c99 struct declarations.
    
    Subclasses can override the following class attributes: 
    
        cname       # the name of the c99 structure
        dtype       # the type definition
        subtypes    # the class definition of nested Blob-types

    todo: use __metaclass__
        
    """

    MAX_DTYPE_PARAM = 5000
    PADDING_FIELD_SUFFIX = '__padding'

    _dtypes = {} # map of known types for reuse and save memory
    @classmethod
    def is_complex(cls):

        if issubclass(cls, BlobArray):
            return True

        for field, subtype in cls.subtypes:
            if issubclass(subtype, Blob) and not issubclass(subtype, BlobEnum):
                return True

        return False

    @classmethod
    def is_plain(cls):

        if issubclass(cls, BlobArray):
            return False

        for field, subtype in cls.subtypes:
            if type(subtype) == type and issubclass(subtype, Blob) and not subtype.is_plain():
                return False

        return True

    @classmethod
    def get_requirements(cls):
        requirements = []
        for field, subtype in cls.subtypes:
            if type(subtype) == type and issubclass(subtype, Blob):
                requirements.extend(subtype.get_requirements())

        if cls.subtypes:
            fields, subtypes = zip(*cls.subtypes)
            requirements.extend(subtypes)
        return requirements

    @classmethod
    def get_dtype_param_keys(cls):
        keys = []

        if issubclass(cls, BlobArray):
            keys.append(BlobArray.CAPACITY_FIELD)
            subparams = cls.child_type.get_dtype_param_keys()
            if subparams:
                for item in subparams:
                    assert item not in keys
                    keys.append(item)
        elif hasattr(cls, 'subtypes'):
            for field, subtype in cls.subtypes:
                if issubclass(subtype, Blob):
                    for item in subtype.get_dtype_param_keys():
                        keys.append('%s_%s' % (field, item))

                else:
                    pass  # Blob without Blob fields
        else:
            pass  # Blob without Blob fields

        return keys

    @classmethod
    def get_dummy_dtype_params(cls):
        dtype_params = {}
        for key in cls.get_dtype_param_keys():
            dtype_params[key] = 1
        return dtype_params

    @classmethod
    @process_dtype_params
    def create_dtype(cls, dtype_params):

        if hasattr(cls, 'dtype'):
            return cls.dtype

        subtypes = []

        dtype_params_keys = dtype_params.keys()
        dtype_params_keys.sort()

        hashable_dtype_params = [repr(cls)]
        for key in dtype_params_keys:
            hashable_dtype_params.append(key)
            hashable_dtype_params.append(dtype_params[key])
        hashable_dtype_params = tuple(hashable_dtype_params)

        if hashable_dtype_params in cls._dtypes:
            return cls._dtypes[hashable_dtype_params]

        for field, subtype in cls.subtypes:
            subtype_params = cls.explode_dtype_params(field=field, dtype_params=dtype_params)
            if issubclass(subtype, Blob):
                subtypes.append((field, subtype.create_dtype(dtype_params=subtype_params)))
            else:
                subtypes.append((field, subtype))

        dtype, subtypes_ = Blob.create_plain_dtype(*subtypes)

        cls._dtypes[hashable_dtype_params] = dtype
        return dtype

    @classmethod
    @process_dtype_params
    def allocate_blob(cls, dtype_params, dtype):
        blob = cls.unshape(blob=numpy.zeros(1, dtype))
        cls.init_blob(blob=blob, dtype_params=dtype_params)
        return dtype, blob

    @classmethod
    def get_subtypes_params(cls):
        subtype_params = {}

        if issubclass(cls, BlobArray):
            subtype_params.update(cls.child_type.get_subtypes_params())
            subtype_params[BlobArray.CAPACITY_FIELD] = BlobArray.CAPACITY_FIELD
        elif hasattr(cls, 'subtypes'):
            for field, subtype in cls.subtypes:
                if issubclass(subtype, Blob):
                    subtype_params[field] = subtype.get_subtypes_params()

        return flat_struct(subtype_params)

    @classmethod
    @validate_dtype_params
    def explode_dtype_params(cls, field, dtype_params):
        field_prefix = '%s_' % field

        subtypes_params = cls.get_subtypes_params()
        subtype_params = {}
        for key, params in subtypes_params.items():
            if key.startswith(field):
                subtype_param_key = key[len(field_prefix):]

                assert key in dtype_params, 'assert %s in %s for subtype of %s' % (key, dtype_params, cls)
                value = dtype_params[key]
                assert isinstance(value, int) or isinstance(value, numpy.int32), '%s should be an integer instead of %s' % (value, type(value))

                subtype_params[subtype_param_key] = value

        return subtype_params

    @classmethod
    @validate_dtype_params
    def sizeof_dtype(cls, dtype_params):
        size = 0

        for field, subtype in cls.subtypes:
            subtype_params = cls.explode_dtype_params(field=field, dtype_params=dtype_params)
            subtype_size = subtype.sizeof_dtype(dtype_params=subtype_params)
            size += subtype_size

        return size

    @classmethod
    @validate_dtype_params
    def explode_blob(cls, blob, dtype_params, field=None):

        offset = 0
        blobs = {}

        if hasattr(cls, 'subtypes'):
            for subtype_field, subtype in cls.subtypes:

                subtype_params = cls.explode_dtype_params(field=subtype_field, dtype_params=dtype_params)
                if filter(lambda item: item is None, subtype_params.values()):
                    subtype_params = subtype.get_dtype_params_from_blob(blob, offset=offset)

                if issubclass(subtype, Blob):
                    subtype_dtype, subtype_blob = subtype.cast_blob(blob=blob, offset=offset, dtype_params=subtype_params)
                    subtype_byte_count = subtype.sizeof_dtype(dtype_params=subtype_params)
                else:
                    subtype_blob = None
                    subtype_byte_count = subtype().itemsize

                if field == subtype_field:
                    return subtype_blob

                blobs[subtype_field] = subtype_blob

                offset += subtype_byte_count
        else:
            raise NotImplementedError()

        if field is not None:
            raise NotImplementedError()

        return blobs

    @classmethod
    @validate_dtype_params
    def init_blob(cls, blob, dtype_params):
        if hasattr(cls, 'subtypes'):
            blobs = cls.explode_blob(blob=blob, dtype_params=dtype_params)

            assert isinstance(blobs, dict), '%s has an invalid explode_blob implementation ' % cls
            assert len(blobs) == len(cls.subtypes), '%s has an invalid explode_blob implementation ' % cls

            for field, subtype in cls.subtypes:
                if issubclass(subtype, Blob):
                    subtype_params = cls.explode_dtype_params(field=field, dtype_params=dtype_params)
                    subtype.init_blob(blob=blobs[field], dtype_params=subtype_params)

    # noinspection PyUnusedLocal
    @classmethod
    @process_dtype_params
    def cast_blob(cls, blob, offset, dtype_params, dtype=None):
        try:
            casted_blob_ = blob.getfield(dtype, offset=offset)
            casted_blob = cls.unshape(casted_blob_)
            return dtype, casted_blob
        except:
            logging.error('failed to cast %s' % cls)
            raise

    @classmethod
    def unshape(cls, blob):
        if blob.shape == (1,):
            return blob[0]
        else:
            return blob

    @classmethod
    def get_subtypes(cls):
        """Returns all required Blob-types (without transitions).
        
        Result is a tuple of two lists, with the attribute names and with the types."""

        if hasattr(cls, 'subtypes') and len(cls.subtypes) > 0:
            return zip(*cls.subtypes)[1]
        else:
            return []

    @classmethod
    def create_plain_dtype(cls, *subtypes):
        if implode_float_n:
            return cls._create_aligned_dtype(*subtypes)
        else:
            return cls._create_unaligned_dtype(*subtypes)

    @classmethod
    def _create_unaligned_dtype(cls, *subtypes):

        requirements = []
        dtype_components = []
        for index, component in enumerate(subtypes):
            field, subtype = component

            if numpy.issctype(subtype):
                if hasattr(subtype, 'descr'):
                    for name, sub_dtype in subtype.descr:
                        subfield = '%s_%s' % (field, name)
                        dtype_components.append((subfield, sub_dtype))
                else:
                    dtype_components.append(component)

            elif issubclass(subtype, BlobEnum):
                dtype_components.append((field, BlobEnum.dtype))

            elif issubclass(subtype, Blob) and subtype.is_plain():
                if hasattr(subtype, 'dtype'):
                    sub_dtype = subtype.dtype

                else:
                    sub_dtype, subtype_requirements = subtype.create_plain_dtype(*subtype.subtypes)

                for name, sub_dtype in sub_dtype.descr:
                    subfield = '%s_%s' % (field, name)
                    dtype_components.append((subfield, sub_dtype))

            else:
                raise NotImplementedError()

            if type(subtype) is type and issubclass(subtype, Blob):
                requirements.append(component)

        return numpy.dtype(dtype_components), requirements

    @classmethod
    def _create_aligned_dtype(cls, *subtypes):

        requirements = []

        dtype_components = []
        for index, component in enumerate(subtypes):
            dtype_components.append(component)  #todo: add components of subfields of type blob
            field, subtype = component

            if type(subtype) is type and issubclass(subtype, Blob):
                requirements.append(component)

            # determine padding attributes
            is_last_of_vector = False
            prefix, suffix = '_'.join(field.split('_')[:-1]), field.split('_')[-1]
            vector_index = None
            for vector_field in vector_fields:
                if suffix in vector_field:
                    vector_index = vector_field.index(suffix)

                    if vector_index == len(vector_field) - 1:
                        is_last_of_vector = True
                    elif index < len(subtypes) - 1 and subtypes[index + 1][0].endswith(vector_field[vector_index + 1]):
                        is_last_of_vector = False
                    else:
                        is_last_of_vector = True

            # add padding
            if is_last_of_vector and vector_index == 2:
                dtype_components.append(('_%s%s' % (field, cls.PADDING_FIELD_SUFFIX), subtype))

        return numpy.dtype(dtype_components), requirements

    @classmethod
    def from_struct(cls, struct, blob):
        """Creates and initializes a object from a struct."""

        assert blob.shape == (), 'the blob must be plain'

        self = cls(blob)

        assert hasattr(self, '_init_blob_from_struct'), '%s requires an implementation of _init_blob_from_struct' % cls
        self._init_blob_from_struct(struct=flat_struct(struct), blob=blob)

        return self

    @classmethod
    def get_field_by_param_key(cls, key, parent_field):

        for field, subtype in cls.subtypes:
            parent_field_ = '%s_%s' % (parent_field, field)

            if key.startswith(parent_field_):
                subkey = subtype.get_field_by_param_key(key, parent_field_)

                if key == subkey:
                    return key

            else:
                continue

            raise NotImplementedError()
        raise NotImplementedError()

    @classmethod
    def get_dtype_params_from_blob(cls, blob):

        dtype_param_keys = cls.get_dtype_param_keys()
        dtype_params = cls.get_dummy_dtype_params()
        done_keys = []

        for field, subtype in cls.subtypes:
            for key in dtype_param_keys:
                if key not in done_keys and key.startswith(field):
                    done_keys.append(key)

                    dummy_dtype, dummy_blob = cls.cast_blob(blob=blob, offset=0, dtype_params=dtype_params)
                    key_field = subtype.get_field_by_param_key(key, field)
                    index = get_blob_index(dummy_dtype, key_field)
                    assert index is not None, 'dtype has no field %s' % key_field
                    value = dummy_blob[index]

                    if value > cls.MAX_DTYPE_PARAM:
                        raise BlobValidationException(
                            'dtype param %s must be smaller than %d (%d)' % (key, cls.MAX_DTYPE_PARAM, value))

                    if value < 1:
                        raise BlobValidationException(
                            'dtype param %s must be positive (%d)' % (key, value))

                    dtype_params[key] = value

        assert dtype_param_keys == done_keys, '%s = %s' % (dtype_param_keys, done_keys)

        return dtype_params

    @classmethod
    def from_blob(cls, blob):

        dtype_params = cls.get_dtype_params_from_blob(blob)

        if blob.dtype.kind in ['V', 'S']: # is void
            dtype, blob = cls.cast_blob(blob=blob, offset=0, dtype_params=dtype_params)

        else:
            dtype = cls.create_dtype(dtype_params=dtype_params)

        assert blob.dtype == dtype, diff_dtype(blob.dtype, dtype)

        try:
            return cls(blob=blob, dtype_params=dtype_params)

        except Exception as ex:
            logging.warn('%s.%s', cls, ex)
            raise

    @property
    def blob(self):
        """Space of binary memory (numpy.ndarray)."""

        return self._blob

    @property
    def dtype_params(self):
        result = {}
        for key in self.get_dtype_param_keys():
            result[key] = self.__getattribute__(key)

        return result

    def __init__(self, blob=None, dtype_params=None, dtype=None, **fields):
        """Initializes the object."""
        cls = type(self)

        if blob is None:
            dtype, blob = cls.allocate_blob(dtype_params=dtype_params, dtype=dtype)

        dtype_is_static = False
        if hasattr(self, 'dtype') and isinstance(self.dtype, numpy.dtype):
            dtype_is_static = True

        assert dtype_params or dtype or dtype_is_static, '%s: a dtype or the factory parameter must be set.' % type(self)

        # determine dtype by creation parameters if dtype is unset
        if dtype is None:
            if dtype_is_static:
                dtype = type(self).dtype

            else:
                try:
                    dtype = type(self).create_dtype(dtype_params=dtype_params)

                except Exception:
                    raise

        assert dtype is not None

        # determine blob properties
        blob_properties = zip(*dtype.descr)
        assert len(blob_properties) == 2
        assert len(blob_properties[0]) > 1, 'a Blob must encapsulate more than one variable: %s' % blob_properties

        # init object
        self._blob_fields_, self._blob_field_offsets = blob_properties
        self._blob = blob
        self.dtype = dtype

        # init subtypes
        if hasattr(self, 'subtypes') and len(self.subtypes) > 0:
            blobs = self.explode_blob(blob=blob, dtype_params=dtype_params)
            for subtype_field, subtype_blob in blobs.items():
                if subtype_field in fields:
                    setattr(self, subtype_field, fields.pop(subtype_field))
                    continue

                for name_, type_ in self.subtypes:
                    if subtype_field == name_:
                        subtype_cls = type_

                if issubclass(subtype_cls, Blob) and not issubclass(subtype_cls, BlobEnum):
                    setattr(self, subtype_field, subtype_cls.from_blob(subtype_blob))

        # init plain fields
        for field, value in fields.items():
            setattr(self, field, value)

        assert self.dtype == blob.dtype, diff_dtype(self.dtype, blob.dtype)

    def _init_blob_from_struct(self, struct, blob):
        """Copy all elements of a struct into the blob."""
        assert blob.dtype == self.dtype
        cls = type(self)

        struct = flat_struct(struct)

        # copy elements 
        for name, value in struct.items():
            name = camel_case_to_underscore(name)

            assert name in self._blob_fields_, 'unknown key %s in %s' % (name, self)

            try:
                self.__setattr__(name, value)

            except:
                raise

        # assert that all elements are initialized
        for name, _name in map(lambda name_: (name_, underscore_to_camel_case(name_)), self._blob_fields_):
            assert name in struct or _name in struct or name.endswith(cls.PADDING_FIELD_SUFFIX),  'uninitialized key %s/%s in %s' % (name, _name, type(self))

    def _data_property_type(self, name):
        """Get a function that casts the blob element to the correct python type."""
        property_index = self._blob_fields_.index(name)
        byte_offset = self._blob_field_offsets[property_index]

        def cast_bool(value):
            return bool(value > 0)

        return {
            '<f4': float,
            '<i4': int,
            '|i1': cast_bool
        }[byte_offset]

    def __setattr__(self, name, value):
        """Set the attribute 'name'.
        
        Try to save it in the blob first, then try to save it in the python object.
        """
        if not name in ['_blob_fields_', '_blob_field_offsets', 'dtype'] \
            and name in self._blob_fields_:
            try:
                self._blob[name] = value
            except:
                raise
        else:
            object.__setattr__(self, name, value)

    def __eq__(self, other):
        """
        compares the representation of each field and return false, if one differs.

        repr is used to support NAN values
        """
        for field in self._blob_fields_:
            if repr(getattr(self, field)) != repr(getattr(other, field)):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __diff__(self, other):
        for field in self._blob_fields_:
            if repr(getattr(self, field)) != repr(getattr(other, field)):
                return 'self.%s: %s != %s: other.%s' % (field, getattr(self, field), field, getattr(other, field))
        return True

    def __getattribute__(self, name):
        """Get the value of the attribute 'name'.
        
        Try to get it from the blob first, then try to get it from the python object.
        """
        if not name in ['_blob_fields_', '_blob_field_offsets', 'dtype'] \
            and name in self._blob_fields_:
            value = self._blob[name]
            cast = self._data_property_type(name)

            if value.shape != ():
                value = value[0]

            try:
                return cast(value)
            except:
                raise

        try:
            return object.__getattribute__(self, name)
        except:
            raise

    def to_struct(self):
        """"Generates a struct from the blob data.
        
        @todo: unflat the structure
        """
        struct = {}
        for name in self._blob_fields_:
            struct[underscore_to_camel_case(name)] = self.__getattribute__(name)
        return struct


class BlobArray(Blob):
    """Encapsulates an array of Blob.
    
    A BlobArray is an abstract class that encapsulates an array of Blob objects.
    The number of elements must be known when the object is created.
    It generates a (dummy) c99 struct declaration as well as a deserializer function, which returns a reference to the array.
    
    The subclass can override the following class attributes:
    
        cname          # the name of the c99 structure
        
    It must implement the following:
        
        child_type     # the Blob-type of the array elements
        
    """

    child_type = None  # must be overridden by specialized class

    STATIC_FIELDS_BYTES = 8
    CAPACITY_FIELD = 'capacity'
    COUNT_FIELD_NAME = 'count'
    INDEX_FIELD = 'global_index'
    dtype_static_components = [
        (CAPACITY_FIELD, numpy.int32),
        (COUNT_FIELD_NAME, numpy.int32)
    ]

    _dtypes = {} # map of known types for reuse and save memory

    @classmethod
    def get_item_field(cls, index, name):
        return 'item_%d_%s' % (index, name)

    @classmethod
    @process_dtype_params
    def create_dtype(cls, dtype_params):
        """Creates the dtype based on the number of items."""

        child_dtype, capacity = cls.create_child_dtype(dtype_params)

        dtype_params_keys = dtype_params.keys()
        dtype_params_keys.sort()

        hashable_dtype_params = [repr(cls)]
        for key in dtype_params_keys:
            hashable_dtype_params.append(key)
            hashable_dtype_params.append(dtype_params[key])
        hashable_dtype_params = tuple(hashable_dtype_params)

        if hashable_dtype_params in cls._dtypes:
            return cls._dtypes[hashable_dtype_params]

        assert child_dtype.names[0] == 'global_index', 'child_type of %s has no global_index field' % cls
        assert isinstance(capacity, int) or isinstance(capacity, numpy.int32), 'expected int instead of %s: %s' % (
        type(capacity), capacity)
        capacity = int(capacity)

        dtype_components = cls.dtype_static_components[:]

        assert capacity < 1000000

        for index in xrange(capacity):
            for name, dtype in child_dtype.descr:
                dtype_components.append((cls.get_item_field(index, name), dtype))

        dtype = numpy.dtype(dtype_components)

        cls._dtypes[hashable_dtype_params] = dtype

        return dtype

    @classmethod
    def get_subtypes(cls):
        """Returns all direct required Blob-types."""
        return [cls.child_type]

    @classmethod
    def get_requirements(cls):
        """Returns all recursive required Blob-types."""
        requirements = cls.child_type.get_requirements()
        requirements.append(cls.child_type)
        return requirements

    @classmethod
    @validate_dtype_params
    def create_child_dtype(cls, dtype_params):
        dtype_params = dtype_params.copy()
        capacity = dtype_params.pop(cls.CAPACITY_FIELD)

        assert hasattr(cls, 'child_type'), '%s, %s' % (repr(cls), dtype_params)

        if hasattr(cls.child_type, 'dtype'):
            child_dtype = cls.child_type.dtype
        else:
            child_dtype = cls.child_type.create_dtype(dtype_params=dtype_params)

        return child_dtype, capacity

    @classmethod
    @validate_dtype_params
    def sizeof_dtype(cls, dtype_params):
        """Calculates the memory space, which the object requires."""

        child_dtype, capacity = cls.create_child_dtype(dtype_params=dtype_params)

        metadata_size = numpy.dtype(cls.dtype_static_components).itemsize
        content_size = child_dtype.itemsize * capacity

        return metadata_size + content_size

    @classmethod
    @process_dtype_params
    def init_blob(cls, blob, dtype_params, dtype=None):

        assert blob.dtype == dtype, diff_dtype(blob.dtype, dtype)

        child_dtype, capacity = cls.create_child_dtype(dtype_params)

        #set initial values
        blob[get_blob_index(dtype, cls.CAPACITY_FIELD)] = capacity
        blob[get_blob_index(dtype, cls.COUNT_FIELD_NAME)] = 0

        for index in range(capacity):
            item_blob = cls.get_item_blob(blob=blob, index=index, child_dtype=child_dtype)
            cls.child_type.init_blob(blob=item_blob, dtype_params=dtype_params)
            item_blob[get_blob_index(child_dtype, cls.INDEX_FIELD)] = -1


    @classmethod
    def get_item_blob(cls, blob, index, child_dtype=None):
        """Returns a blob of the element with an index."""
        assert blob.shape == (), 'exptected blob shape %s' % blob.shape
        assert isinstance(index, int), 'the index must be in integer'

        if child_dtype is None and len(cls.child_type.get_dtype_param_keys()) == 0:
            child_dtype = cls.child_type.dtype

        # get offset
        offset = cls.STATIC_FIELDS_BYTES
        offset += child_dtype.itemsize * index

        assert cls.STATIC_FIELDS_BYTES + child_dtype.itemsize * index + child_dtype.itemsize <= blob.size * blob.itemsize, \
            '%s: dtype:%d + %d * %d + %d = %d <= blob:%d * %d = %d' % (
                repr(cls),
                cls.STATIC_FIELDS_BYTES, child_dtype.itemsize,
                index, child_dtype.itemsize,
                (cls.STATIC_FIELDS_BYTES + child_dtype.itemsize * index + child_dtype.itemsize),
                blob.size, blob.itemsize,
                blob.nbytes
            )

        # get blob
        item_blob = blob.getfield(child_dtype, offset=offset)

        # validate blob
        assert item_blob.shape == (), 'blob must be a flat struct instead of %s' % str(item_blob.shape)

        # return blob
        return item_blob

    @classmethod
    def from_blob(cls, blob):

        dtype_params = cls.get_dtype_params_from_blob(blob=blob)
        dtype, casted_blob = cls.cast_blob(blob=blob, offset=0, dtype_params=dtype_params)

        try:
            return cls(blob=casted_blob, dtype_params=dtype_params, capacity=dtype_params[cls.CAPACITY_FIELD])
        except Exception as ex:
            logging.warn('%s.%s', cls, ex)
            raise

    @classmethod
    def from_struct(cls, struct, blob):
        """Creates and initializes a object from a struct."""

        assert blob.shape == (), 'the blob must be plain'

        items = []
        for index, child_struct in enumerate(struct):
            child_blob = cls.get_item_blob(blob, index, child_dtype=None)
            items.append(cls.child_type.from_struct(child_struct, child_blob))

        self = cls(blob, items=items)

        return self

    @classmethod
    def get_field_by_param_key(cls, key, parent_field):
        key_ = '%s_%s' % (parent_field, cls.CAPACITY_FIELD)
        if key == key_:
            return key

        key_ = key[len(parent_field + '_'):]

        key_ = '%s_%s' % (parent_field, cls.get_item_field(index=0, name=key_))
        return key_

    @classmethod
    def validate_capacity(cls, capacity):

        isinstance(capacity, int), 'dtype_params must contain the integer field capacity'

        if capacity <= 0:
            raise BlobValidationException('array capacity must be positive instead of %d' % capacity)

        if capacity > Blob.MAX_DTYPE_PARAM:
            raise BlobValidationException('array capacity must smaller than 1000 instead of %d' % capacity)

    @classmethod
    def get_dtype_params_from_blob(cls, blob):

        offset = 0

        # create dummy
        dtype_params = cls.get_dummy_dtype_params()
        dummy_dtype, dummy_blob = cls.cast_blob(blob=blob, offset=offset, dtype_params=dtype_params)

        # determine capacity
        field_index = get_blob_index(dummy_dtype, cls.CAPACITY_FIELD)
        capacity = dummy_blob[field_index]
        cls.validate_capacity(capacity)
        dtype_params[cls.CAPACITY_FIELD] = capacity

        child_dtype, capacity = cls.create_child_dtype(dtype_params)
        item_blob = cls.get_item_blob(blob=dummy_blob, index=0, child_dtype=child_dtype)
        child_dtype_params = cls.child_type.get_dtype_params_from_blob(blob=item_blob)

        dtype_params.update(child_dtype_params)

        return dtype_params

    def __init__(self, blob, dtype_params=None, dtype=None, items=None, capacity=None):
        cls = type(self)

        if items is None:
            items = []

        while len(items) < capacity: #todo: remove this later
            items.append(None)

        if capacity is None:
            capacity = len(items)

        assert capacity > 0, '%s: an empty list? use java for that!' % cls
        assert blob.shape == ()
        assert blob.nbytes > 4, '%s: the blob must be greater than the index variable space' % cls

        if dtype_params is None:
            dtype_params = {cls.CAPACITY_FIELD: capacity}

        assert cls.CAPACITY_FIELD in dtype_params
        cls.validate_capacity(dtype_params[cls.CAPACITY_FIELD])

        Blob.__init__(self, blob=blob, dtype=dtype, dtype_params=dtype_params)

        self._items = items

        if self.capacity != capacity:
            self.capacity = capacity

        valid_item_count = len(filter(lambda item: item is not None, items))
        if self.count == 0 and valid_item_count:
            for index in range(self.capacity - 1, 0 - 1, -1):
                if items[index] is not None:
                    self.count = index + 1
                    break

        else:
            child_dtype, capacity_ = self.create_child_dtype(dtype_params)

            count = 0
            for index in range(self.capacity):
                item_blob = self.get_item_blob(blob=blob, index=index, child_dtype=child_dtype)
                index_ = item_blob[get_blob_index(item_blob.dtype, 'global_index')]
                if index_ > -1:
                    self._items[index] = self.child_type.from_blob(item_blob)
                    count += 1

            try:
                if self.count != count:
                    self.count = count

            except RuntimeError:
                raise

    def __getitem__(self, index):
        """Returns the element at the given index."""
        return self._items[index]

    def __iter__(self):
        """Returns an iterator over the stored elements."""
        return self._items.__iter__()

    def __len__(self):
        """Returns the number of stored elements."""
        return self.count

    def to_struct(self):
        """Returns a struct representation of all stored elements."""
        return [item.to_struct() for item in self._items]


class BlobLinkedList(BlobArray):

    def next_blob(self):
        self.count += 1
        index = self.count - 1
        assert index < self.capacity, 'not enough capacity %d < %d' % (index, self.capacity)

        item_blob = BlobArray.get_item_blob(blob=self.blob, index=index, child_dtype=self.child_type.dtype)
        global_index_field_index = get_blob_index(self.child_type.dtype, 'global_index')
        item_blob[global_index_field_index] = index
        return item_blob, index

    def append(self, item):
        assert isinstance(item, self.child_type)
        assert item.global_index > -1
        assert item.global_index == self.count - 1
        self._items[item.global_index] = item


class BlobEnum(Blob):

    dtype, subtypes = numpy.int32, []

    UNDEFINED = 'undefined'

    @classmethod
    def string_to_int(cls, string):
        for index, field in enumerate(cls.fields):
            if string == field:
                return index
        raise NotImplementedError(string)

    @classmethod
    def int_to_string(cls, index, ignore_errors=False):
        if index <= 0:
            if ignore_errors:
                return cls.UNDEFINED

            else:
                raise EnumException()

        elif index >= len(cls.fields):
            if ignore_errors:
                return '%d is out of range' % index

            else:
                raise EnumException()

        else:
            return cls.fields[index]
