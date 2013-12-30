
import logging
import numpy
import itertools

from blob_types.utils import flat_struct, camel_case_to_underscore, underscore_to_camel_case, get_blob_index, dtype_to_lines

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
    
    @classmethod
    def get_subtypes(cls):
        """Returns all required Blob-types (without transitions).
        
        Result is a tuple of two lists, with the attribute names and with the types."""
        
        if hasattr(cls, 'subtypes'):
            return zip(*cls.subtypes)[1]
        else:
            return []

    @classmethod
    def create_flat_dtype(cls, subtypes):
        """Creates a flat dtype structure that stores all elements including that of nested types."""

        dtype_components = []
        for sub_name, sub_dtype in subtypes:
            if type(sub_dtype) is type:
                dtype_components.append((sub_name, sub_dtype))
            elif type(sub_dtype) is numpy.dtype:
                for name, dtype in sub_dtype.descr:
                    dtype_components.append(('%s_%s' % (sub_name, name), dtype))
            else:
                raise NotImplementedError('flat dtype of %s : %s : %s' % (sub_name, type(sub_dtype), sub_dtype))
        return numpy.dtype(dtype_components)
    
    @classmethod
    def from_struct(cls, struct, blob):
        """Creates and initializes a object from a struct."""
        
        assert blob.shape == (), 'the blob must be plain'
        
        self = cls(blob)

        assert hasattr(self, '_init_blob_from_struct'), '%s requires an implementation of _init_blob_from_struct' % cls

        struct = flat_struct(struct)
        self._init_blob_from_struct(struct, blob)
        return self

    @classmethod
    def from_blob(cls, blob):
        return cls(blob)
    
    @property
    def blob(self):
        """Space of binary memory (numpy.ndarray)."""
        
        return self._blob
    
    def __init__(self, blob, dtype_params=None, dtype=None, validate_blob=True):
        """Initializes the object."""

        if dtype_params is None:
            dtype_params = {}
        
        dtype_is_static = hasattr(type(self), 'dtype') and isinstance(type(self).dtype, numpy.dtype)

        assert dtype_params or dtype or dtype_is_static, 'a dtype or the factory parameter must be set.'
        
        # determine dtype by creation parameters if dtype is unset
        if dtype is None:
            if dtype_is_static:
                dtype = type(self).dtype
                #logging.info('get dtype from class %s', type(self))
            else:
                try:
                    dtype = type(self).create_dtype(**dtype_params)
                    #logging.info('get dtype for %s from params %s', type(self), dtype_params)
                except Exception:
                    raise
        assert dtype is not None
        
        # determine blob properties
        blob_properties = zip(*dtype.descr)
        assert len(blob_properties) == 2
        assert len(blob_properties[0]) > 1, 'a Blob must encapsulate more than one variable: %s' % blob_properties
        
        # init object
        self._blob_property_names, self._blob_property_offsets = blob_properties
        self._blob = blob
        self.dtype = dtype

        if self.dtype != blob.dtype:
            # this is verbose debug output, and only printed if something in the source code is really wrong!

            radius = 3
            unaligned = 0
            count = 0

            for dtype_field, blob_dtype_field in itertools.izip_longest(dtype_to_lines(self.dtype), dtype_to_lines(blob.dtype)):
                if (dtype_field != blob_dtype_field or unaligned > 0) and unaligned <= radius:
                    logging.debug('%s: %s # %s', count, dtype_field, blob_dtype_field)
                    unaligned += 1
                count += 1

            logging.debug('dtype_is_static = %s', dtype_is_static)
            logging.debug('dtype_params = %s', dtype_params)
            logging.debug('dtype is None = %s', dtype is None)

        assert self.dtype == blob.dtype, 'invalid dtype or blob'# (dtype_params %s or dtype %s or dtype_is_static %s)' % (dtype_params is not None, self.dtype is not None, dtype_is_static)
        
        # validate object
        if validate_blob:
            self.validate_blob()
        
    def validate_blob(self):
        """Validates the blob and dtype."""
        
        assert self.dtype, 'a valid dtype must be available'
        
        blob_keys = zip(*self.blob.dtype.descr)[0]
        keys = zip(*self.dtype.descr)[0]
        
        # compare own dtype and blob dtype 
        for index in range(len(blob_keys)):
            area = keys[index]
            try:
                area = ']['.join([keys[index - 1], keys[index], keys[index + 1]])
            except IndexError:
                pass
            assert keys[index] == blob_keys[index], \
                'blob attribute #%d: %s is unexpected at position of %s' % (index, blob_keys[index], area)
  
    def _init_blob_from_struct(self, struct, blob):
        """Copy all elements of a struct into the blob."""
        assert blob.dtype == self.dtype

        struct = flat_struct(struct)
        
        # copy elements 
        for name, value in struct.items():
            name = camel_case_to_underscore(name)

            assert name in self._blob_property_names, 'unknown key %s in %s' % (name, self)

            try:
                self.__setattr__(name, value)
            except:
                raise
        
        # assert that all elements are initialized
        for name, _name in map(lambda name_: (name_, underscore_to_camel_case(name_)), self._blob_property_names):
            assert name in struct.keys() or _name in struct.keys(), 'uninitialized key %s/%s in %s' % (name, _name, type(self))
        
    def _data_property_type(self, name):
        """Get a function that casts the blob element to the correct python type."""
        property_index = self._blob_property_names.index(name)
        byte_offset = self._blob_property_offsets[property_index]

        def cast_bool(value):
            return bool(value > 0)

        return {'<f4': float, '<i4': int, '|i1': cast_bool}[byte_offset]
        
    def __setattr__(self, name, value):
        """Set the attribute 'name'.
        
        Try to save it in the blob first, then try to save it in the python object.
        """
        if not name in ['_blob_property_names', '_blob_property_offsets', 'dtype'] \
                and name in self._blob_property_names:
            try:
                self._blob[name] = value
            except:
                raise
        else:
            object.__setattr__(self, name, value)
    
    def __getattribute__(self, name):
        """Get the value of the attribute 'name'.
        
        Try to get it from the blob first, then try to get it from the python object.
        """
        if not name in ['_blob_property_names', '_blob_property_offsets', 'dtype'] \
                and name in self._blob_property_names:
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
        for name in self._blob_property_names:
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
    
    STATIC_FIELDS_BYTES = 8
    CAPACITY_FIELD_NAME = 'capacity'
    COUNT_FIELD_NAME = '_count'
    dtype_static_components = [
        (CAPACITY_FIELD_NAME,   numpy.int32),
        (COUNT_FIELD_NAME,      numpy.int32)
    ]

    _dtypes = {} # map of known types for reuse and save memory

    @classmethod
    def create_dtype(cls, capacity, child_dtype=None):
        """Creates the dtype based on the number of items."""
        
        if child_dtype is None:
            child_dtype = cls.child_type.dtype

        dtype_params = (capacity, child_dtype)

        if dtype_params in cls._dtypes:
            return cls._dtypes[dtype_params]

        assert child_dtype.names[0] == 'global_index', 'child_type of %s has no global_index field' % cls
        assert isinstance(capacity, int) or isinstance(capacity, numpy.int32), 'expected int instead of %s' % type(capacity)
        capacity = int(capacity)
        
        dtype_components = cls.dtype_static_components[:]

        assert capacity < 1000000

        for index in xrange(capacity):
            for name, dtype in child_dtype.descr:
                dtype_components.append(('item_%d_%s' % (index, name), dtype))
            
        dtype = numpy.dtype(dtype_components)

        cls._dtypes[dtype_params] = dtype

        return dtype
    
    @classmethod
    def get_subtypes(cls):
        """Returns all direct required Blob-types."""
        return [cls.child_type]

    @classmethod
    def sizeof_dtype(cls, capacity, child_dtype=None):
        """Calculates the memory space, which the object requires."""

        if child_dtype is None:
            child_dtype = cls.child_type.dtype
        
        size = numpy.dtype(cls.dtype_static_components).itemsize
        size += child_dtype.itemsize * capacity
        return size

    @classmethod
    def allocate_blob(cls, capacity, child_dtype=None):
        dtype = cls.create_dtype(capacity=capacity, child_dtype=None)
        blob = numpy.zeros(1, dtype)[0]

        if child_dtype is None:
            child_dtype = cls.child_type.dtype

        assert blob.nbytes == cls.sizeof_dtype(capacity=capacity, child_dtype=child_dtype)

        cls.empty_blob(blob, capacity, child_dtype)

        logging.debug('array allocated %d kb for %d items' % (blob.nbytes / 1024, capacity))
        return dtype, blob

    @classmethod
    def empty_blob(cls, blob, capacity, child_dtype=None):

        if child_dtype is None:
            child_dtype = cls.child_type.dtype

        for index in range(capacity):
            item_blob = cls.get_item_blob(blob, index, child_dtype)
            item_blob[get_blob_index(child_dtype, 'global_index')] = -1

    
    @classmethod
    def get_item_blob(cls, blob, index, child_dtype = None):
        """Returns a blob of the element with an index."""
        assert blob.shape == (), 'exptected blob shape %s' % blob.shape
        assert isinstance(index, int), 'the index must be in integer'
        
        if child_dtype is None:
            child_dtype = cls.child_type.dtype
        
        # get offset
        offset = cls.STATIC_FIELDS_BYTES
        offset += child_dtype.itemsize * index

        assert cls.STATIC_FIELDS_BYTES + child_dtype.itemsize * index + child_dtype.itemsize <= blob.size * blob.itemsize,\
            'dtype:%d + %d * %d + %d = %d <= blob:%d * %d = %d' % (
                cls.STATIC_FIELDS_BYTES, child_dtype.itemsize, index, child_dtype.itemsize,
                (cls.STATIC_FIELDS_BYTES + child_dtype.itemsize * index + child_dtype.itemsize), # dtypesize
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
        capacity = blob.getfield(numpy.dtype(int), offset=0) # 0 because capacity is the first field

        if isinstance(capacity, numpy.ndarray):
            capacity = capacity[0]

        assert isinstance(capacity, int), 'capactity should be an integer instead of %s' % type(capacity)
        self = cls(blob=blob, capacity=capacity)
        return self
        
    def __init__(self, blob, dtype_params = None, items = None, capacity = None):

        #logging.info("%s: %s, %s", type(self), item_count, type(items))

        if dtype_params is None:
            dtype_params = {}

        if items is None:
            items = []
        while len(items) < capacity: #todo: remove this later
            items.append(None)

        if capacity is None:
            capacity = len(items)

        #logging.info("%s: %d, %s", type(self), item_count, type(items))

        assert capacity > 0, 'an empty list? use java for that!'
        assert blob.shape == ()
        assert blob.nbytes > 4, 'the blob must be greater than the index variable space'

        if not dtype_params:
            dtype_params = {'capacity': capacity}
        assert 'capacity' in dtype_params and isinstance(dtype_params['capacity'], int), 'dtype_params must contain the integer field capacity'
        
        Blob.__init__(self, blob=blob, dtype_params=dtype_params)#, blob_properties=zip(*self.dtype.descr)
        
        self._items = items
        self.capacity = capacity

        valid_item_count = len(filter(lambda item: item is not None, items))
        if self._count == 0 and valid_item_count:
            for index in range(self.capacity - 1, 0 - 1, -1):
                if items[index] is not None:
                    self._count = index + 1
                    break
            logging.info('init count from item list to %d', self._count)
        else:
            self._count = 0
            for index in range(self.capacity):
                item_blob = self.get_item_blob(blob, index)
                index_ = item_blob[get_blob_index(item_blob.dtype, 'global_index')]
                if index_ > -1:
                    self._items[index] = self.child_type.from_blob(item_blob)
                    self._count += 1
            logging.info('init count from blob items to %d', self._count)
        
    def __getitem__(self, index):
        """Returns the element at the given index."""
        return self._items[index]
    
    def __iter__(self):
        """Returns an iterator over the stored elements."""
        return self._items.__iter__()
    
    def __len__(self):
        """Returns the number of stored elements."""
        return self._count

    def get_item_blob_(self, index):
        return self.get_item_blob(self.blob, index)
    
    def to_struct(self):
        """Returns a struct representation of all stored elements."""
        return [item.to_struct() for item in self._items]

