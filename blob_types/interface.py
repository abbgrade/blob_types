import logging
import numpy
import pyopencl as opencl
from pyopencl.compyte.dtypes import dtype_to_ctype

from blob_types.types import Blob, BlobArray
from blob_types.utils import camel_case_to_underscore, implode_floatn


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

    blob_types.extend(filter(lambda blob_type_: issubclass(blob_type_, Blob), dependencies))
    return blob_types


class BlobInterface(object):
    def __init__(self, blob_type):
        assert issubclass(blob_type, Blob)
        self.blob_type = blob_type

    def get_cname(self, address_space_qualifier):
        """Returns the type name in c99-style including a address space qualifier (local, constant, global, private)."""

        if hasattr(self.blob_type, 'cname'):
            cname = self.blob_type.cname
        else:
            cname = '%s_t' % camel_case_to_underscore(self.blob_type.__name__)

        return ('%s %s' % (address_space_qualifier, cname)).strip()

    def get_sizeof_cname(self, address_space_qualifier):
        """Returns the c99 function name of the sizeof function."""

        return 'sizeof_%s_%s' % (address_space_qualifier[0], self.get_cname(''))

    def get_deserialize_cname(self, address_space_qualifier):
        """Returns the function name of the c99 deserializer function."""

        return 'deserialize_%s_%s' % (address_space_qualifier[0], self.get_cname(''))

    def get_init_cname(self, address_space_qualifier):
        """Returns the c99 init function name."""

        return 'init_%s_%s' % (address_space_qualifier[0], self.get_cname(''))

    def get_ctype(self, dtype=None):
        """Returns the c99 declaration of the type."""

        if dtype is None:
            dtype = self.blob_type.dtype

        field_definitions = []
        for field in dtype.names:
            field_definitions.append('\t%s %s;' % (dtype_to_ctype(dtype.fields[field][0]),  field))
        field_definitions = '\n'.join(field_definitions)

        definition = \
'''
typedef struct {
%(fields)s
} %(cname)s;
''' % {
    'fields': field_definitions,
    'cname': self.get_cname('')
}
        return definition.strip()


    def get_csizeof(self, address_space_qualifier):
        """Creates a c99 sizeof method."""

        definition = 'unsigned long %(function_name)s(%(fullcname)s* blob)' % {
            'function_name': self.get_sizeof_cname(address_space_qualifier),
            'fullcname': self.get_cname(address_space_qualifier)
        }

        if hasattr(self.blob_type, 'dtype') and isinstance(self.blob_type.dtype, numpy.dtype):
            # flat structs or primitive types can use sizeof(type)
            declaration = \
'''
%(definition)s
{
    return sizeof(%(cname)s);
};
''' % {
    'definition': definition,
    'cname': self.get_cname('')
}

        else:
            # complex types must calculate the size by the size of its components.
            arguments = ['blob'] # the first argument must be the data itself.

            lines = [] # all required source code lines
            variables = [] # all required variable names

            # iterate over all components/subtypes
            for field, dtype in self.blob_type.subtypes:
                field_variable = '%s_instance' % field

                if numpy.issctype(dtype):
                    # determine the size of the scalar type
                    cname = dtype_to_ctype(dtype)
                    variables.append('%s %s* %s;' % (address_space_qualifier, cname, field_variable))
                    sizeof_call = 'sizeof(%s)' % cname
                else:
                    # determine the size of the complex type
                    assert issubclass(dtype, Blob), 'unexpected type %s %s' % (type(dtype), dtype)
                    variables.append(
                        '%s* %s;' % (BlobLib.get_interface(dtype).get_cname(address_space_qualifier), field_variable))
                    sizeof_call = '%s(%s)' % (
                    BlobLib.get_interface(dtype).get_sizeof_cname(address_space_qualifier), field_variable)

                # save which arguments and lines are required to determine the total size
                arguments.append('&%s' % field_variable)
                lines.append('size += %s;' % sizeof_call)

            lines.insert(0, '%s(%s);' % (self.get_deserialize_cname(address_space_qualifier), ', '.join(arguments)))

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
    'cname': self.get_cname(address_space_qualifier),
    'lines': '\n'.join(['\t' + line for line in lines])
}
        return definition.strip() + ';', declaration.strip()

    def get_cdeserializer(self, address_space_qualifier):
        """Returns the c99 deserializer function declaration, which separates the components of a flat type."""

        arguments = ['%s* blob' % self.get_cname(address_space_qualifier)]
        declarations = []
        lines = []
        previous_field_offset, previous_field_space = 0, 0

        last_field = self.blob_type.subtypes[-1][0]

        # iterate over all subtypes/components
        for field, dtype in self.blob_type.subtypes:
            is_last_field = field == last_field

            # format
            lines.append('')
            lines.append('/* cast of %s */' % field)

            # used variable names
            field_variable = '%s_instance' % field
            field_offset = '%s_offset' % field
            if not is_last_field:
                field_space = '%s_space' % field

            declarations.append('unsigned long %s;' % field_offset)
            if not is_last_field:
                declarations.append('unsigned long %s;' % field_space)

            # add sizeof call of component
            if numpy.issctype(dtype):
                cname = "%s %s" % (address_space_qualifier, dtype_to_ctype(dtype))
                sizeof_call = 'sizeof(%s)' % cname
            else:
                assert issubclass(dtype, Blob), 'unexpected type %s %s' % (type(dtype), dtype)
                cname = BlobLib.get_interface(dtype).get_cname(address_space_qualifier)
                sizeof_call = '%s(*%s)' % (
                BlobLib.get_interface(dtype).get_sizeof_cname(address_space_qualifier), field_variable)

            # add component reference to arguments (for results)
            arguments.append('%s** %s' % (cname, field_variable))

            # determine offset of component
            lines.append('%s = %s + %s;' % (field_offset, previous_field_offset, previous_field_space))

            # set and cast component reference
            lines.append(
                '*%s = (%s*)(((%s char*)blob) + %s);' % (field_variable, cname, address_space_qualifier, field_offset))

            if not is_last_field:
                # determine size of component
                lines.append('%s = %s;' % (field_space, sizeof_call))

            previous_field_space = field_space
            previous_field_offset = field_offset

        lines = ['\t' + line for line in lines]

        definition = 'void %s(%s)' % (self.get_deserialize_cname(address_space_qualifier), ', '.join(arguments))
        # fill function template
        lines.insert(0, definition)
        lines.insert(1, '{')
        for index, line in enumerate(declarations):
            lines.insert(2 + index, '\t' + line)
        lines.append('}')
        declaration = '\n'.join(lines)

        return definition.strip() + ';', declaration

    def get_init_ctype(self, *args, **kwargs):
        """Returns the c99 init function declaration."""

        assert 'address_space_qualifier' in kwargs, 'keyword address_space_qualifier is required'

        # initializer of a constant type must be a dummy, so ignore the constant specifier.
        if kwargs['address_space_qualifier'] == 'constant':
            cname = self.get_cname('')
        else:
            cname = self.get_cname(kwargs['address_space_qualifier'])

        # add initialization of all components, which contain a _item_count component
        lines = []
        dtype = self.blob_type.create_dtype(dtype_params=self.blob_type.get_dummy_dtype_params())
        for index, field in enumerate(filter(lambda field_: field_.endswith('_item_count'), dtype.names)):
            lines.append('blob->%s = %d;' % (field, args[index]))

        # fill the function template
        definition = 'void %(function_name)s(%(type)s* blob)' % {
            'function_name': self.get_init_cname(kwargs['address_space_qualifier']),
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


class BlobArrayInterface(BlobInterface):

    FIRST_ITEM_FIELD = 'first_item'

    def __init__(self, blob_type):
        assert issubclass(blob_type, BlobArray)
        BlobInterface.__init__(self, blob_type)

    def get_ctype(self, dtype=None):
        """Returns the c99 struct declaration."""

        field_definitions = []
        for field, subdtype in self.blob_type.dtype_static_components:
            field_definitions.append('\t%s %s;' % (dtype_to_ctype(subdtype),  field))
        field_definitions = '\n'.join(field_definitions)

        return \
'''typedef struct {
%(static_fields)s
    %(child_cname)s %(first_item_field)s;
} %(cname)s;''' % {
    'static_fields': field_definitions,
    'first_item_field' : self.FIRST_ITEM_FIELD,
    'child_cname': BlobLib.get_interface(self.blob_type.child_type).get_cname(''),
    'cname': self.get_cname('')
}

    def get_csizeof(self, address_space_qualifier):
        definition = 'unsigned long %(function_name)s(%(cname)s* list)' % {
            'function_name': self.get_sizeof_cname(address_space_qualifier),
            'cname': self.get_cname(address_space_qualifier)
        }

        declaration = \
'''
%(definition)s
{
    unsigned long static_fields_space = %(static_fields_space)s;
    unsigned long items_space = list->%(capacity_field)s * %(child_sizeof_cname)s(&(list->%(first_item_field)s));
    return static_fields_space + items_space;
};
''' % {
    'definition': definition.strip(),
    'static_fields_space': BlobArray.STATIC_FIELDS_BYTES,
    'capacity_field': BlobArray.CAPACITY_FIELD,
    'first_item_field': self.FIRST_ITEM_FIELD,
    'child_sizeof_cname': BlobLib.get_interface(self.blob_type.child_type).get_sizeof_cname(
        address_space_qualifier),
    'child_cname': BlobLib.get_interface(self.blob_type.child_type).get_cname(address_space_qualifier)
}
        return definition.strip() + ';', declaration.strip()

    def get_cdeserializer(self, address_space_qualifier):
        """Returns the c99 deserializer function."""
        definition = 'void %(function_name)s(%(cname)s* list, %(child_cname)s** array)' % {
            'function_name': self.get_deserialize_cname(address_space_qualifier),
            'child_cname': BlobLib.get_interface(self.blob_type.child_type).get_cname(address_space_qualifier),
            'cname': self.get_cname(address_space_qualifier)}
        declaration = \
'''
%(definition)s {
    array[0] = &(list->%(first_item_field)s);
};''' % {
    'definition' : definition,
    'first_item_field' : self.FIRST_ITEM_FIELD
}
        return definition + ';', declaration


class BlobLib(object):
    """Generates C-code which allows to work with the serialized blob data."""

    @classmethod
    def get_interface(cls, blob_type):
        assert issubclass(blob_type, Blob)
        if issubclass(blob_type, BlobArray):
            return BlobArrayInterface(blob_type)
        else:
            return BlobInterface(blob_type)

    def __init__(self,
                 device=None,
                 required_constant_blob_types=None,
                 required_global_blob_types=None,
                 header_header='',
                 header_footer=''):

        if device is None:
            # noinspection PyUnusedLocal
            device = opencl.create_some_context(False).get_info(opencl.context_info.DEVICES)[0]

        if required_constant_blob_types is None:
            required_constant_blob_types = []

        if required_global_blob_types is None:
            required_global_blob_types = []

        self.type_definitions = []
        self.function_definitions = []
        self.function_declarations = []

        for address_space_qualifier, required_blob_types in [('constant', required_constant_blob_types),
                                                             ('global', required_global_blob_types)]:

            # check dependencies
            blob_types = walk_dependencies(required_blob_types)

            # generate header
            generated_types = []
            for blob_type in blob_types:
                # ingnore duplicates
                if blob_type in generated_types:
                    continue
                else:
                    generated_types.append(blob_type)

                blob_type_interface = BlobLib.get_interface(blob_type)

                if issubclass(blob_type, BlobArray):
                    # The type is an array: add dummy type and deserializer.

                    try:
                        type_declaration = blob_type_interface.get_ctype()
                        if type_declaration not in self.type_definitions:
                            self.type_definitions.append(type_declaration)

                        definition, declaration = blob_type_interface.get_cdeserializer(address_space_qualifier)
                        self.function_definitions.append(definition)
                        self.function_declarations.append(declaration)

                        definition, declaration = blob_type_interface.get_csizeof(address_space_qualifier)
                        self.function_definitions.append(definition)
                        self.function_declarations.append(declaration)
                    except:
                        raise

                elif not hasattr(blob_type, 'dtype'):
                    # The type has components of variable length: add dummy and deserializer.
                    try:
                        # try to generate a dummy .... hehe ... look at this dirty approach
                        create_dtype_params, param_count = [], 0

                        dtype = blob_type.create_dtype(dtype_params=blob_type.get_dummy_dtype_params())
                        type_declaration = implode_floatn(blob_type_interface.get_ctype(dtype=dtype))

                        if type_declaration == '':
                            # check if the dummy creation was successful
                            logging.warn('unable to generate dummy ctype %s' % blob_type_interface.get_cname(
                                address_space_qualifier))
                            raise
                        else:
                            if type_declaration not in self.type_definitions:
                                self.type_definitions.append(type_declaration)

                            # add a dummy initializer function
                            try:
                                definition, declaration = blob_type_interface.get_init_ctype(
                                    address_space_qualifier=address_space_qualifier)
                                self.function_definitions.append(definition)
                                self.function_declarations.append(declaration)
                            except:
                                logging.warn('unable to generate dummy initializer ctype %s with %d params' % (
                                blob_type_interface.get_cname(address_space_qualifier), param_count))
                                raise
                    except:
                        raise

                    definition, declaration = blob_type_interface.get_cdeserializer(address_space_qualifier)
                    self.function_definitions.append(definition)
                    self.function_declarations.append(declaration)

                    definition, declaration = blob_type_interface.get_csizeof(address_space_qualifier)
                    self.function_definitions.append(definition)
                    self.function_declarations.append(declaration)

                else:
                    # The type has a constant size ... add the c type declaration.
                    type_declaration = implode_floatn(blob_type_interface.get_ctype())
                    if type_declaration not in self.type_definitions:
                        self.type_definitions.append(type_declaration)

                    definition, declaration = blob_type_interface.get_csizeof(address_space_qualifier)
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
