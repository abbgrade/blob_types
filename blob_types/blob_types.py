'''
Created on 26.09.2013

@author: skampm2s
'''
import logging
import numpy

import pyopencl as opencl
from pyopencl.compyte.dtypes import dtype_to_ctype

# utils

def camelCaseToUnderscore(name):
    """Convert a CamelCaseString into an underscore_string."""
    
    new_name = []
    for char in name:
        if char.isupper() and len(new_name) > 0:
            new_name.append('_' + char.lower())
        else:
            new_name.append(char.lower())
    return ''.join(new_name)

def underscoreToCamelCase(name):
    """Convert a underscore_string into a CamelCaseString."""
    
    name = ''.join(map(lambda part: part.capitalize(), name.split('_')))
    name = name[0].lower() + name[1:]
    return name

def flat_struct(struct):
    """Copy a struct, which contains all values but without nesting dicts of lists.
    
    @example:
        {
            'foo': 'bar', 
            'pos': {
                'x': 4,
                'y': 2
            },
            'lorem': [
                'ipsum', 
                1, 
                2, 
                3
            ]
        }
        
        will be:
        
        {
            'foo': 'bar', 
            'pos_x': 4,
            'pos_y': 2,
            'lorem_0':'ipsum', 
            'lorem_1': 1, 
            'lorem_2': 2, 
            'lorem_3': 3
        }
    """
    
    result = {}
    for key, value in struct.items():
        if isinstance(value, dict):
            for subkey, value in flat_struct(value).items():
                result['%s_%s' % (camelCaseToUnderscore(key), subkey)] = value
        elif isinstance(value, list):
            for index, value in enumerate(value):
                result['%s_%d' % (camelCaseToUnderscore(key), index)] = value
        else:
            result[camelCaseToUnderscore(key)] = value
    return result

def implode_floatn(struct_source):
    """Convert vector types."""
    lines = []
    for line in struct_source.split('\n'):
        parts = line.split()
        if len(parts) == 2 and parts[0] == 'float' and parts[1].endswith('_y;') and lines[-1][:-2] == line[:-2]:
            lines[-1] = line.replace('float ', 'float2 ')[:-3] + ';'
        else:
            lines.append(line)
    return '\n'.join(lines)

def walk_dependencies(dependencies):
    """Determine the dependencies recursively."""
    
    blob_types = []
    for blob_type in dependencies:
        if blob_type not in blob_types:
            if issubclass(blob_type, Blob):
                blob_types.extend(walk_dependencies(blob_type.get_subtypes()))
            elif numpy.issctype(blob_type):
                pass
            else:
                assert False, 'blob_type must be a subclass of blob_types.Blob or a numpy.dtype'
            
    blob_types.extend(filter(lambda blob_type: issubclass(blob_type, Blob), dependencies))
    return blob_types
            

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
    def get_cname(cls, address_space_qualifier):
        """Returns the type name in c99-style including a address space qualifier (local, constant, global, private)."""
        
        if hasattr(cls, 'cname'):
            cname = cls.cname
        else:
            cname = '%s_t' % camelCaseToUnderscore(cls.__name__)
        
        return ('%s %s' % (address_space_qualifier, cname)).strip()
        
    @classmethod
    def get_ctype(cls, dtype=None):
        """Returns the c99 declaration of the type."""
        
        if dtype is None:
            dtype = cls.dtype
            
        definition = \
'''
typedef struct {
%(fields)s
} %(cname)s;
''' % {'fields': '\n'.join(['\t%(type)s %(name)s;' % {'name': field, 'type': dtype_to_ctype(dtype.fields[field][0])}
                            for field in dtype.names]), 
       'cname': cls.get_cname('')
}
        return definition.strip()

    
    @classmethod
    def get_csizeof(cls, address_space_qualifier):
        """Creates a c99 sizeof method."""
        
        definition = 'unsigned long %(function_name)s(%(fullcname)s* blob)' % {
            'function_name': cls.get_sizeof_cname(address_space_qualifier),
            'fullcname': cls.get_cname(address_space_qualifier)
        } 
        
        if hasattr(cls, 'dtype') and isinstance(cls.dtype, numpy.dtype):
            # flat structs or primitive types can use sizeof(type)
            declaration = \
'''
%(definition)s
{
    return sizeof(%(cname)s);
};
''' % {
    'definition': definition,
    'cname': cls.get_cname('')
}

        else:
            # complex types must calculate the size by the size of its components.
            arguments = ['blob'] # the first argument must be the data itself.
            
            lines = [] # all required source code lines
            variables = [] # all required variable names
            
            # iterate over all components/subtypes
            for field, dtype in cls.subtypes:
                field_variable = '%s_instance' % field
                
                if numpy.issctype(dtype):
                    # determine the size of the scalar type
                    cname = dtype_to_ctype(dtype)
                    variables.append('%s %s* %s;' % (address_space_qualifier, cname, field_variable))
                    sizeof_call = 'sizeof(%s)' % cname
                else:
                    # determine the size of the complex type
                    assert issubclass(dtype, Blob), 'unexpected type %s %s' % (type(dtype), dtype)
                    variables.append('%s* %s;' % (dtype.get_cname(address_space_qualifier), field_variable))
                    sizeof_call = '%s(%s)' % (dtype.get_sizeof_cname(address_space_qualifier), field_variable)
                    
                # save which arguments and lines are required to determine the total size
                arguments.append('&%s' % field_variable)
                lines.append('size += %s;' % sizeof_call)
                
            lines.insert(0, '%s(%s);' % (cls.get_deserialize_cname(address_space_qualifier), ', '.join(arguments)))
            
            # prepend the variable declarations to the source code                
            variables.extend(lines)
            lines = variables
            
            # fill the function template
            declaration = \
'''
%(definition)s
{
    unsigned long size = 0;
%(lines)s
    return size;
}
''' % {
    'definition': definition.strip(),
    'cname': cls.get_cname(address_space_qualifier), 
    'lines': '\n'.join(['\t' + line for line in lines])
}
        return definition.strip() + ';', declaration.strip()

    @classmethod
    def get_sizeof_cname(cls, address_space_qualifier):
        """Returns the c99 function name of the sizeof function."""
        
        return 'sizeof_%s_%s' % (address_space_qualifier[0], cls.get_cname(''))

    @classmethod
    def get_cdeserializer(cls, address_space_qualifier):
        """Returns the c99 deserializer function declaration, which separates the components of a flat type."""
        
        arguments = ['%s* blob' % cls.get_cname(address_space_qualifier)]
        declarations = []
        lines = []
        previous_field_offset, previous_field_space = 0, 0
        
        # iterate over all subtypes/components
        for field, dtype in cls.subtypes:
            # format
            lines.append(    '')
            lines.append(    '/* cast of %s */' % field)
            
            # used variable names
            field_variable = '%s_instance' % field
            field_offset =   '%s_offset' % field
            field_space =    '%s_space' % field
            
            declarations.append('unsigned long %s;' % field_offset)
            declarations.append('unsigned long %s;' % field_space)
            
            # add sizeof call of component
            cname = None
            if numpy.issctype(dtype):
                cname = "%s %s" % (address_space_qualifier, dtype_to_ctype(dtype))
                sizeof_call = 'sizeof(%s)' % cname
            else:
                assert issubclass(dtype, Blob), 'unexpected type %s %s' % (type(dtype), dtype)
                cname = dtype.get_cname(address_space_qualifier)
                sizeof_call = '%s(*%s)' % (dtype.get_sizeof_cname(address_space_qualifier), field_variable)
            
            # add component reference to arguments (for results)
            arguments.append('%s** %s' % (cname, field_variable))
            
            # determine offset of component
            lines.append('%s = %s + %s;' % (field_offset, previous_field_offset, previous_field_space))
            
            # set and cast component reference 
            lines.append('*%s = (%s*)(((%s char*)blob) + %s);' % (field_variable, cname, address_space_qualifier, field_offset))
            
            # determine size of component
            lines.append('%s = %s;' % (field_space, sizeof_call))
            
            previous_field_space = field_space
            previous_field_offset = field_offset
        
        lines = ['\t' + line for line in lines]
        
        definition = 'void %s(%s)' % (cls.get_deserialize_cname(address_space_qualifier), ', '.join(arguments))
        # fill function template
        lines.insert(0, definition)
        lines.insert(1, '{')
        for index, line in enumerate(declarations):
            lines.insert(2 + index, '\t' + line) 
        lines.append('}')
        declaration = '\n'.join(lines)
        
        return definition.strip() + ';', declaration
    
    @classmethod
    def get_deserialize_cname(cls, address_space_qualifier):
        """Returns the function name of the c99 deserializer function."""
        
        return 'deserialize_%s_%s' % (address_space_qualifier[0], cls.get_cname(''))
    
    @classmethod
    def get_init_ctype(cls, *args, **kwargs):
        """Returns the c99 init function declaration."""
        
        assert 'address_space_qualifier' in kwargs, 'keyword address_space_qualifier is required'
        
        # initializer of a constant type must be a dummy, so ignore the constant specifier.
        if kwargs['address_space_qualifier'] == 'constant':
            cname = cls.get_cname('')
        else:
            cname = cls.get_cname(kwargs['address_space_qualifier'])
        
        # add initialization of all components, which contain a _item_count component
        lines = []
        dtype = cls.create_dtype(*args)
        for index, field in enumerate(filter(lambda field: field.endswith('_item_count'), dtype.names)):
            lines.append('blob->%s = %d;' % (field, args[index]))
        
        # fill the function template
        definition = 'void %(function_name)s(%(type)s* blob)' % {
            'function_name': cls.get_init_cname(kwargs['address_space_qualifier']),
            'type': cname
        }
        declaration = \
'''
%(definition)s
{
%(content)s
};''' % {
            'definition': definition.strip(),
            'content': '\n'.join(['\t' + line for line in lines])
        }
        return definition.strip() + ';', declaration
    
    @classmethod
    def get_init_cname(cls, address_space_qualifier):
        """Returns the c99 init function name."""
        
        return 'init_%s_%s' % (address_space_qualifier[0], cls.get_cname(''))
 
    @classmethod
    def create_flat_dtype(cls, subtypes):
        """Creates a flat dtype structure that stores all elements including that of nested types."""
        
        dtype_components = []
        for sub_name, sub_dtype in subtypes:
            if type(sub_dtype) is type:
                dtype_components.append((sub_name, sub_dtype))
            else:
                for name, dtype in sub_dtype.descr:
                    dtype_components.append(('%s_%s' % (sub_name, name), dtype))
        return numpy.dtype(dtype_components)
    
    @classmethod
    def from_struct(cls, struct, blob):
        """Creates and initializes a object from a struct."""
        
        assert blob.shape == (), 'the blob must be plain'
        
        self = cls(blob)
        self._init_blob_from_struct(flat_struct(struct), blob)
        return self
    
    @property
    def blob(self):
        """Space of binary memory (numpy.ndarray)."""
        
        return self._blob
    
    def __init__(self, blob, dtype_params={}, dtype=None, validate_blob=True):
        """Initializes the object."""
        
        dtype_is_static = hasattr(type(self), 'dtype') and isinstance(type(self).dtype, numpy.dtype)
        assert dtype_params or dtype or dtype_is_static, 'a dtype or the factory parameter must be set.'
        
        # determine dtype by creation parameters if dtype is unset
        if dtype is None:
            if dtype_is_static:
                dtype = type(self).dtype
            else:
                try:
                    dtype = type(self).create_dtype(**dtype_params)
                except Exception:
                    raise
        assert dtype is not None
        
        # determine blob proberties
        blob_properties = zip(*dtype.descr)
        assert len(blob_properties) == 2
        assert len(blob_properties[0]) > 1, 'a Blob must encapsulate more than one variable: %s' % blob_properties
        
        # init object
        self._blob_property_names, self._blob_property_offsets = blob_properties
        self._blob = blob
        self.dtype = dtype
        assert self.dtype == blob.dtype, 'invalid dtype or blob'
        
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
            except:
                pass
            assert keys[index] == blob_keys[index], 'blob attribute #%d: %s is unexpected at position of %s' % (index, blob_keys[index], area) 
  
    def _init_blob_from_struct(self, struct, blob):
        """Copy all elements of a struct into the blob."""
        
        # copy elements 
        for name, value in struct.items():
            name = camelCaseToUnderscore(name)
            try:
                self.__setattr__(name, value)
            except:
                logging.warn('unknown key %s in %s' % (name, self))
                raise
        
        # assert that all elements are initialized
        for name, _name in map(lambda name: (name, underscoreToCamelCase(name)), self._blob_property_names):
            assert (name in struct.keys() or _name in struct.keys()), 'uninitialized key %s/%s in %s' % (name, _name, type(self))
        
    def _data_property_type(self, name):
        """Get a function that casts the blob element to the correct python type."""
        property_index = self._blob_property_names.index(name)
        byte_offset = self._blob_property_offsets[property_index]
        return {'<f4': float,
                '<i4': int}[byte_offset]
        
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
            struct[underscoreToCamelCase(name)] = self.__getattribute__(name)
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
    
    COUNT_ITEM_LENGTH = 4
    dtype_static_components = [('item_count', numpy.int32)]
    
    @classmethod
    def create_dtype(cls, item_count, child_dtype=None):
        """Creates the dtype based on the number of items."""
        
        if child_dtype is None:
            child_dtype=cls.child_type.dtype
        
        dtype_components = cls.dtype_static_components[:]
        
        for index in range(item_count):
            for name, dtype in child_dtype.descr:
                dtype_components.append(('item_%d_%s' % (index, name), dtype))
            
        return numpy.dtype(dtype_components)
    
    @classmethod
    def get_subtypes(cls):
        """Returns all direct required Blob-types."""
        return [cls.child_type]
    
    @classmethod
    def get_ctype(cls):
        """Returns the c99 struct declaration."""
        return \
'''typedef struct {
    int item_count;
    %(child_cname)s first_item;
} %(cname)s;''' % {'child_cname': cls.child_type.get_cname(''), 'cname': cls.get_cname('')}

    @classmethod
    def get_cdeserializer(cls, address_space_qualifier):
        """Returns the c99 deserializer function."""
        definition = 'void %(function_name)s(%(cname)s* list, %(child_cname)s** array)' % {
            'function_name': cls.get_deserialize_cname(address_space_qualifier),
            'child_cname': cls.child_type.get_cname(address_space_qualifier), 
            'cname': cls.get_cname(address_space_qualifier)}
        declaration = \
'''
%s {
    array[0] = &(list->first_item);
};''' % definition
        return definition + ';', declaration

    @classmethod
    def sizeof_dtype(cls, item_count, child_dtype=None):
        """Calculates the memory space, which the object requires."""
        
        if child_dtype is None:
            child_dtype = cls.child_type.dtype
        
        size = numpy.dtype(cls.dtype_static_components).itemsize
        size += child_dtype.itemsize * item_count
        return size
    
    @classmethod
    def get_csizeof(cls, address_space_qualifier):
        definition = 'unsigned long %(function_name)s(%(cname)s* list)' % {
            'function_name': cls.get_sizeof_cname(address_space_qualifier),
            'cname': cls.get_cname(address_space_qualifier)
        }
        declaration = \
'''
%(definition)s
{
    unsigned long count_space = sizeof(int); /* item_count */
    unsigned long items_space = list->item_count * %(child_sizeof_cname)s(&(list->first_item));
    return count_space + items_space;
};
''' % {
            'definition': definition.strip(),
            'child_sizeof_cname': cls.child_type.get_sizeof_cname(address_space_qualifier),
            'child_cname': cls.child_type.get_cname(address_space_qualifier)
        }
        return definition.strip() + ';', declaration.strip()
    
    @classmethod
    def get_item_blob(cls, blob, index, child_dtype = None):
        """Returns a blob of the element with an index."""
        assert blob.shape == (), 'exptected blob shape %s' % blob.shape
        
        assert isinstance(index, int), 'the index must be in integer'
        
        if child_dtype is None:
            child_dtype = cls.child_type.dtype
        
        # get offset
        offset = cls.COUNT_ITEM_LENGTH
        offset += child_dtype.itemsize * index
        
        # get blob
        item_blob = blob.getfield(child_dtype, offset=offset)
        
        # validate blob
        assert item_blob.shape == (), 'blob must be a flat struct instead of %s' % str(item_blob.shape) 
        
        # return blob
        return item_blob
        
    def __init__(self, blob, dtype_params = {}, items = [], item_count = None):
        if item_count is None:
            item_count = len(items)
        assert item_count > 0, 'an empty list? use java for that!'
        assert blob.nbytes > 4, 'the blob must be greater than the index variable space'
        
        if not dtype_params:
            dtype_params = {'item_count': item_count}
        
        Blob.__init__(self, blob=blob, dtype_params=dtype_params)#, blob_properties=zip(*self.dtype.descr)
        
        self._items = items
        self.item_count = len(items)
        
    def __getitem__(self, index):
        """Returns the element at the given index."""
        return self._items[index]
    
    def __iter__(self):
        """Returns an iterator over the stored elements."""
        return self._items.__iter__()
    
    def __len__(self):
        """Returns the number of stored elements."""
        return self.item_count
    
    def to_struct(self):
        """Returns a struct representation of all stored elements."""
        return [item.to_struct() for item in self._items]


class BlobInterface(object):
    
    """Generates C-code which allows to work with the serialized blob data."""
    
    def __init__(self,
            device=None,
            required_constant_blob_types = [],
            required_global_blob_types = [],
            header_header = '',
            header_footer = ''):
        if device is None:
            device = opencl.create_some_context(False).get_info(opencl.context_info.DEVICES)[0]

        self.type_definitions = []
        self.function_definitions = []
        self.function_declarations = []
        
        for address_space_qualifier, required_blob_types in [('constant', required_constant_blob_types), ('global', required_global_blob_types)]:
            
            # check dependencies             
            blob_types = walk_dependencies(required_blob_types)
            logging.debug('required %s types: %s' % (address_space_qualifier, ', '.join([blob_type.get_cname('') for blob_type in blob_types])))
            
            # generate header
            generated_types = []
            for blob_type in blob_types:
                # ingnore duplicates
                if blob_type in generated_types:
                    continue
                else:
                    generated_types.append(blob_type)
                    
                #declarations.append('/* definition of %s */' % blob_type.__name__)
                if issubclass(blob_type, BlobArray):
                    # The type is an array: add dummpy type and deserializer.
                    try:
                        type_declaration = blob_type.get_ctype()
                        if type_declaration not in self.type_definitions:
                            self.type_definitions.append(type_declaration)
                            
                        definition, declaration = blob_type.get_cdeserializer(address_space_qualifier)
                        self.function_definitions.append(definition)
                        self.function_declarations.append(declaration)
                        
                        definition, declaration = blob_type.get_csizeof(address_space_qualifier)
                        self.function_definitions.append(definition)
                        self.function_declarations.append(declaration)
                    except:
                        raise
                    
                elif not hasattr(blob_type, 'dtype'):
                    # The type has components of variable length: add dummy and deserializer.
                    try:
                        # try to generate a dummy .... hehe ... look at this dirty approach
                        type_declaration = ''
                        for param_count in range(25): # try functions with up to 25 params
                            create_dtype_params = [1]*param_count
                            try:
                                dtype = blob_type.create_dtype(*create_dtype_params)
                            except: 
                                continue
                            else:
                                type_declaration = implode_floatn(blob_type.get_ctype(dtype))
                                break
                            
                        if type_declaration == '':
                            # check if the dummy creation was successful
                            logging.warn('unable to generate dummy ctype %s' % blob_type.get_cname(address_space_qualifier))
                            raise
                        else:
                            if type_declaration not in self.type_definitions:
                                self.type_definitions.append(type_declaration)
                            
                            # add a dummy initializer function
                            try:
                                definition, declaration = blob_type.get_init_ctype(*create_dtype_params, address_space_qualifier=address_space_qualifier)
                                self.function_definitions.append(definition)
                                self.function_declarations.append(declaration)
                            except:
                                logging.warn('unable to generate dummy initializer ctype %s with %d params' % (blob_type.get_cname(address_space_qualifier), param_count))
                                raise
                    except:
                        raise
                    
                    definition, declaration = blob_type.get_cdeserializer(address_space_qualifier)
                    self.function_definitions.append(definition)
                    self.function_declarations.append(declaration)
                        
                    definition, declaration = blob_type.get_csizeof(address_space_qualifier)
                    self.function_definitions.append(definition)
                    self.function_declarations.append(declaration)
                
                else:
                    # The type has a constant size ... add the c type declaration.
                    type_declaration = implode_floatn(blob_type.get_ctype())
                    if type_declaration not in self.type_definitions:
                        self.type_definitions.append(type_declaration)
                        
                    definition, declaration = blob_type.get_csizeof(address_space_qualifier)
                    self.function_definitions.append(definition)
                    self.function_declarations.append(declaration)
        
        self.header_code = '\n\n'.join([
            '/* header generated by %s */' % __file__,
            header_header,
            '\n\n'.join(map(str.strip, self.type_definitions)),
            '\n\n'.join(map(str.strip, self.function_definitions)),
            header_footer
        ])
        self.source_code = '\n\n'.join([
            '/* source generated by %s */' % __file__,
            '\n\n'.join(map(str.strip, self.function_declarations))
        ])
