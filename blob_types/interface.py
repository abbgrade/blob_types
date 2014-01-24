"""
... is a submodule of [blob_types](__init__.html).
It contains classes which generate c structs and functions for access to blob_types based data structures.
"""

import numpy
import pyopencl as opencl
from pyopencl.compyte.dtypes import dtype_to_ctype

from types import Blob, BlobArray, BlobEnum
from utils import camel_case_to_underscore, implode_floatn, implode_float_n


class Lib(object):
    @property
    def header_code(self):
        header_code = '\n\n'.join([dependency.header_code for dependency in self.dependecies])

        if self.header_code_path:
            with open(self.header_code_path) as file_handle:
                header_code += '\n\n' + file_handle.read()

        return header_code

    @property
    def source_code(self):
        source_code = '\n\n'.join([dependency.source_code for dependency in self.dependecies])

        if self.source_code_path:
            with open(self.source_code_path) as file_handle:
                source_code += '\n\n' + file_handle.read()

        return source_code

    def __init__(self, dependencies=[], code_path=None):
        self.dependecies = dependencies

        if code_path:
            self.header_code_path = '%s.h' % code_path
            self.source_code_path = '%s.cl' % code_path

        else:
            self.header_code_path = None
            self.source_code_path = None


class BlobInterface(object):
    def __init__(self, blob_type):
        assert issubclass(blob_type, Blob)
        self.blob_type = blob_type

    def get_functions(self, address_space_qualifier):
        raise NotImplementedError('abstract get_cfunctions %s' % type(self))

    def get_address_space_suffix(self, address_space_qualifier):
        return address_space_qualifier[0]

    def get_name(self, address_space_qualifier, clean=False):
        """Returns the type name in c99-style including a address space qualifier (local, constant, global, private)."""

        cname = camel_case_to_underscore(self.blob_type.__name__)

        if clean:
            return cname

        else:
            cname = '%s_%st' % (cname, self.get_address_space_suffix(address_space_qualifier))
            return cname

    def get_spaced_name(self, address_space_qualifier):

        return '%s %s' % (address_space_qualifier, self.get_name(address_space_qualifier))

    def get_sizeof_name(self, address_space_qualifier):
        """Returns the c99 function name of the sizeof function."""

        return 'sizeof_%s' % self.get_name(address_space_qualifier)

    def get_deserialize_name(self, address_space_qualifier):
        """Returns the function name of the c99 deserializer function."""

        return 'deserialize_%s' % self.get_name(address_space_qualifier)

    def get_init_name(self, address_space_qualifier):
        """Returns the c99 init function name."""

        return 'init_%s_%s' % (
            self.get_address_space_suffix(address_space_qualifier),
            self.get_name(address_space_qualifier)
        )

    def get_type(self, address_space_qualifier):
        """Returns the c99 declaration of the type."""

        raise NotImplementedError('abstract get_type %s' % type(self))

    def get_sizeof(self, address_space_qualifier):
        """Creates a c99 sizeof method."""

        raise NotImplementedError('abstract get_sizeof %s' % type(self))


class BlobComplexInterface(BlobInterface):

    def get_functions(self, address_space_qualifier):

        return zip(*[
            self.get_sizeof(address_space_qualifier),
            self.get_deserialize(address_space_qualifier)
        ])

    def get_type(self, address_space_qualifier):
        """Returns the c99 deserializer function declaration, which separates the components of a flat type."""

        fields = []

        # iterate over all subtypes/components
        for field, subtype in self.blob_type.subtypes:
            if field.endswith(Blob.PADDING_FIELD_SUFFIX):
                continue

            # used variable names

            # add sizeof call of component
            if numpy.issctype(subtype):
                fields.append('%s %s* %s;' % (address_space_qualifier, dtype_to_ctype(subtype), field))

            else:
                assert issubclass(subtype, Blob), 'unexpected type %s %s' % (type(subtype), subtype)
                if subtype.is_plain():
                    fields.append('%s* %s;' % (
                        BlobLib.get_interface(subtype).get_spaced_name(address_space_qualifier),
                        field
                    ))

                else:
                    fields.append('%s %s;' % (
                        BlobLib.get_interface(subtype).get_name(address_space_qualifier),
                        field
                    ))

        definition = \
'''
/* complex type %(name)s */

typedef struct _%(name)s
{
    %(fields)s
} %(name)s;
''' % {
    'name': self.get_name(address_space_qualifier, ),
    'fields': '\n\t'.join(fields)
}
        return definition

    def get_sizeof(self, address_space_qualifier):
        """Creates a c99 sizeof method."""

        definition = 'unsigned long %(function_name)s(%(address_space_qualifier)s char* blob)' % {
            'function_name': self.get_sizeof_name(address_space_qualifier),
            'address_space_qualifier': address_space_qualifier,
        }

        arguments = ['blob', '&self']  # the first argument must be the data itself.
        variables = ['%s %s;' % (self.get_name(address_space_qualifier, ), 'self')]  # all required variable names
        lines = []  # all required source code lines

        # iterate over all components/subtypes
        for field, subtype in self.blob_type.subtypes:
            if field.endswith(Blob.PADDING_FIELD_SUFFIX):
                continue

            if numpy.issctype(subtype):
                # determine the size of the scalar type
                cname = dtype_to_ctype(subtype)
                sizeof_call = 'sizeof(%s)' % cname
            else:
                # determine the size of the complex type
                assert issubclass(subtype, Blob), 'unexpected type %s %s' % (type(subtype), subtype)
                sizeof_call = '%s((%s char*)(blob + size))' % (
                    BlobLib.get_interface(subtype).get_sizeof_name(address_space_qualifier),
                    address_space_qualifier,
                )

            # save which arguments and lines are required to determine the total size
            lines.append('size += %s;' % sizeof_call)

        lines.insert(0, '%s(%s);' % (self.get_deserialize_name(address_space_qualifier), ', '.join(arguments)))

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
    'cname': self.get_name(address_space_qualifier),
    'lines': '\n'.join(['\t' + line for line in lines])
}
        return definition.strip() + ';', declaration.strip()

    def get_deserialize(self, address_space_qualifier):
        """Returns the c99 deserializer function declaration, which separates the components of a flat type."""

        arguments = ['%s char* blob' % address_space_qualifier]
        declarations = []
        lines = []
        previous_field_offset, previous_field_space = 0, 0

        last_field = self.blob_type.subtypes[-1][0]

        # iterate over all subtypes/components
        for field, subtype in self.blob_type.subtypes:
            if field.endswith(Blob.PADDING_FIELD_SUFFIX):
                continue

            is_last_field = field == last_field

            # format
            lines.append('')
            lines.append('/* cast of %s */' % field)

            # used variable names
            field_variable = 'self->%s' % field
            field_offset = '%s_offset' % field
            field_reference = 'blob + %s' % field_offset
            if not is_last_field:
                field_space = '%s_space' % field

            declarations.append('unsigned long %s;' % field_offset)
            if not is_last_field:
                declarations.append('unsigned long %s;' % field_space)

            # add sizeof call of component
            if numpy.issctype(subtype):
                cname = "%s %s" % (address_space_qualifier, dtype_to_ctype(subtype))
                sizeof_call = 'sizeof(%s)' % cname

            else:
                assert issubclass(subtype, Blob), 'unexpected type %s %s' % (type(subtype), subtype)
                cname = "%s %s" % (
                    address_space_qualifier,
                    BlobLib.get_interface(subtype).get_name(address_space_qualifier)
                )
                sizeof_call = '%s((%s char*)%s)' % (
                    BlobLib.get_interface(subtype).get_sizeof_name(address_space_qualifier),
                    address_space_qualifier,
                    field_reference
                )

            # determine offset of component
            lines.append('%s = %s + %s;' % (field_offset, previous_field_offset, previous_field_space))

            # set and cast component reference
            if not numpy.issctype(subtype) and not subtype.is_plain():
                lines.append('%s(%s, &%s);' % (
                    BlobLib.get_interface(subtype).get_deserialize_name(address_space_qualifier),
                    field_reference,
                    field_variable
                ))
            else:
                lines.append('%s = (%s*)(%s);' % (field_variable, cname, field_reference))

            if not is_last_field:
                # determine size of component
                lines.append('%s = %s;' % (field_space, sizeof_call))

            previous_field_space = field_space
            previous_field_offset = field_offset

        lines = ['\t' + line for line in lines]

        arguments.append('%s* %s' % (self.get_name(address_space_qualifier, ), 'self'))

        definition = 'void %s(%s)' % (self.get_deserialize_name(address_space_qualifier), ', '.join(arguments))

        # fill function template
        lines.insert(0, definition)
        lines.insert(1, '{')
        for index, line in enumerate(declarations):
            lines.insert(2 + index, '\t' + line)
        lines.append('}')
        declaration = '\n'.join(lines)

        return definition.strip() + ';', declaration


class BlobPlainInterface(BlobInterface):

    def get_functions(self, address_space_qualifier):
        functions = [self.get_sizeof(address_space_qualifier)]

        for field, dtype in self.blob_type.get_blob_fields(recursive=True):
            if not issubclass(dtype, BlobEnum):
                functions.append(self.get_accessor(field, dtype, address_space_qualifier))

        return zip(*functions)

    def get_type(self, address_space_qualifier):
        """Returns the c99 declaration of the type."""

        field_definitions = []
        for field in self.blob_type.dtype.names:
            if field.endswith(Blob.PADDING_FIELD_SUFFIX):
                continue
            field_definitions.append('\t%s %s;' % (dtype_to_ctype(self.blob_type.dtype.fields[field][0]), field))
        field_definitions = '\n'.join(field_definitions)

        definition = \
'''
/* plain type %(cname)s */

typedef struct __attribute__((__packed__)) _%(cname)s
{
%(fields)s
} %(cname)s;
''' % {
    'fields': field_definitions,
    'cname': self.get_name(address_space_qualifier)
}
        return definition.strip()

    def get_sizeof(self, address_space_qualifier):
        """Creates a c99 sizeof method."""

        definition = 'unsigned long %(function_name)s(%(address_space_qualifier)s char* blob)' % {
            'function_name': self.get_sizeof_name(address_space_qualifier),
            'address_space_qualifier': address_space_qualifier
        }

        declaration = \
'''
%(definition)s
{
    return sizeof(%(name)s);
};
''' % {
    'definition': definition,
    'name': self.get_name(address_space_qualifier),
}
        return definition.strip() + ';', declaration.strip()

    def get_accessor_name(self, field, address_space_qualifier):
        """Returns the c99 function name of the accessor function."""

        return 'get_%s_%s' % (self.get_name(address_space_qualifier), field)

    def get_accessor(self, field, dtype, address_space_qualifier):

        child_name = BlobLib.get_interface(dtype).get_spaced_name(address_space_qualifier)

        definition = '%(child_name)s * %(function_name)s(%(address_space_qualifier)s %(cname)s* self)' % {
            'function_name': self.get_accessor_name(field, address_space_qualifier),
            'cname': self.get_name(address_space_qualifier),
            'child_name': child_name,
            'address_space_qualifier': address_space_qualifier
        }

        field_chain = [field]
        current_dtype = dtype
        while not numpy.issctype(current_dtype):
            subfield, current_dtype = current_dtype.subtypes[0]
            field_chain.append(subfield)

        declaration = \
'''
%(definition)s
{
    return (%(child_name)s *)&self->%(field)s;
};''' % {
    'definition': definition,
    'child_name': child_name,
    'field': '_'.join(field_chain)
}
        return definition + ';', declaration


class BlobEnumInterface(BlobPlainInterface):

    def get_functions(self, address_space_qualifier):
        return [], []

    def get_type(self, address_space_qualifier):

        field_definitions = []
        field_constants = []
        for index, field in enumerate(self.blob_type.fields):
            if len(field) == 0:
                field = BlobEnum.UNDEFINED
            constant = self.get_name(address_space_qualifier, clean=True).upper() + '_' + camel_case_to_underscore(
                field).upper()
            field_definitions.append('\t%s = %s' % (camel_case_to_underscore(field), constant))
            field_constants.append('#define %s %d' % (constant, index))

        return \
'''
/* enum type %(cname)s */

%(field_constants)s

typedef enum _%(cname)s
{
%(static_fields)s
} %(cname)s;
''' % {
    'field_constants': '\n'.join(field_constants),
    'static_fields': ',\n'.join(field_definitions),
    'cname': self.get_name(address_space_qualifier)
}


class BlobArrayInterface(BlobComplexInterface):

    FIRST_ITEM_FIELD = 'first'

    def get_functions(self, address_space_qualifier):
        functions = [
            self.get_sizeof(address_space_qualifier),
            self.get_deserialize(address_space_qualifier),
            self.get_item(address_space_qualifier)
        ]
        return zip(*functions)

    def get_type(self, address_space_qualifier):
        """Returns the c99 struct declaration."""

        field_definitions = []
        for field, subdtype in self.blob_type.dtype_static_components:
            field_definitions.append('\t%s %s* %s;' % (address_space_qualifier, dtype_to_ctype(subdtype), field))

        child_name = BlobLib.get_interface(self.blob_type.child_type).get_spaced_name(address_space_qualifier)

        return \
'''
/* array type %(name)s */

typedef struct __attribute__((__packed__)) _%(name)s
{
%(static_fields)s
    %(address_space_qualifier)s char* %(first_item_field)s;
} %(name)s;''' % {
    'static_fields': '\n'.join(field_definitions),
    'first_item_field': self.FIRST_ITEM_FIELD,
    'child_name': child_name,
    'address_space_qualifier': address_space_qualifier,
    'name': self.get_name(address_space_qualifier)
}

    def get_sizeof(self, address_space_qualifier):
        definition = 'unsigned long %(function_name)s(%(address_space_qualifier)s char* blob)' % {
            'function_name': self.get_sizeof_name(address_space_qualifier),
            'address_space_qualifier': address_space_qualifier,
        }

        declaration = \
'''
%(definition)s
{
    int capacity = *((%(address_space_qualifier)s int*) blob);
    unsigned long static_fields_space = %(static_fields_space)s;
    unsigned long sizeof_child = %(child_sizeof_cname)s(blob + static_fields_space);
    unsigned long items_space = capacity * sizeof_child;
    return static_fields_space + items_space;
};
''' % {
    'definition': definition.strip(),
    'static_fields_space': BlobArray.STATIC_FIELDS_BYTES,
    'capacity_field': BlobArray.CAPACITY_FIELD,
    'first_item_field': self.FIRST_ITEM_FIELD,
    'child_sizeof_cname': BlobLib.get_interface(self.blob_type.child_type).get_sizeof_name(
        address_space_qualifier),
    'child_cname': BlobLib.get_interface(self.blob_type.child_type).get_name(address_space_qualifier),
    'address_space_qualifier': address_space_qualifier,
}
        return definition.strip() + ';', declaration.strip()

    def get_deserialize(self, address_space_qualifier):
        """Returns the c99 deserializer function."""

        definition = 'void %(function_name)s(%(address_space_qualifier)s char* blob, %(cname)s* self)' % {
            'function_name': self.get_deserialize_name(address_space_qualifier),
            'cname': self.get_name(address_space_qualifier),
            'address_space_qualifier': address_space_qualifier
        }
        declaration = \
'''
%(definition)s
{
    self->%(capacity_field)s = (%(address_space_qualifier)s int*)(blob);
    self->%(count_field)s = (%(address_space_qualifier)s int*)(blob + %(static_fields_space)s / 2);
    self->%(first_item_field)s = blob + %(static_fields_space)s;
};''' % {
    'definition': definition,
    'static_fields_space': BlobArray.STATIC_FIELDS_BYTES,
    'capacity_field': BlobArray.CAPACITY_FIELD,
    'count_field': BlobArray.COUNT_FIELD_NAME,
    'first_item_field': self.FIRST_ITEM_FIELD,
    'address_space_qualifier': address_space_qualifier,
}
        return definition + ';', declaration

    def get_item_name(self, address_space_qualifier):
        """Returns the function name of the c99 item function."""

        return 'get_%s_item' % self.get_name(address_space_qualifier)

    def get_item(self, address_space_qualifier):
        """Returns the c99 deserializer function."""
        if self.blob_type.child_type.is_plain():
            return self.get_plain_item(address_space_qualifier)

        else:
            return self.get_complex_item(address_space_qualifier)

    def get_plain_item(self, address_space_qualifier):

        child_name = BlobLib.get_interface(self.blob_type.child_type).get_spaced_name(address_space_qualifier)

        definition = '%(child_name)s * %(function_name)s(%(cname)s array, int index)' % {
            'function_name': self.get_item_name(address_space_qualifier),
            'cname': self.get_name(address_space_qualifier),
            'child_name': child_name
        }

        child_sizeof_name = BlobLib.get_interface(self.blob_type.child_type).get_sizeof_name(address_space_qualifier)

        declaration = \
'''
%(definition)s
{
    if(index < 0)
        return 0;
    if(index >= *array.capacity)
        return 0;

    unsigned long offset = index * %(sizeof_function)s(array.%(first_item_field)s);
    %(address_space_qualifier)s char* item_blob = array.%(first_item_field)s + offset;
    return (%(child_name)s *)item_blob;
};''' % {
    'definition': definition,
    'sizeof_function': child_sizeof_name,
    'first_item_field': self.FIRST_ITEM_FIELD,
    'address_space_qualifier': address_space_qualifier,
    'child_name': child_name
}
        return definition + ';', declaration

    def get_complex_item(self, address_space_qualifier):

        child_name = BlobLib.get_interface(self.blob_type.child_type).get_name(address_space_qualifier)

        definition = 'void %(function_name)s(%(cname)s array, int index, %(child_name)s * item)' % {
            'function_name': self.get_item_name(address_space_qualifier),
            'cname': self.get_name(address_space_qualifier),
            'child_name': child_name
        }

        child_sizeof_name = BlobLib.get_interface(self.blob_type.child_type).get_sizeof_name(address_space_qualifier)
        child_deserialize_name = BlobLib.get_interface(self.blob_type.child_type).get_deserialize_name(
            address_space_qualifier)

        declaration = \
'''
%(definition)s
{
    unsigned long offset = index * %(sizeof_function)s(array.%(first_item_field)s);
    %(address_space_qualifier)s char* item_blob = array.%(first_item_field)s + offset;
    %(child_deserialize_name)s(item_blob, item);
};''' % {
    'definition': definition,
    'sizeof_function': child_sizeof_name,
    'first_item_field': self.FIRST_ITEM_FIELD,
    'address_space_qualifier': address_space_qualifier,
    'child_deserialize_name': child_deserialize_name
}
        return definition + ';', declaration


class BlobLib(Lib):
    """Generates C-code which allows to work with the serialized blob data."""

    @classmethod
    def get_interface(cls, blob_type):
        assert issubclass(blob_type, Blob), 'blob_type=%s must be a subclass of Blob' % blob_type

        if issubclass(blob_type, BlobEnum):
            return BlobEnumInterface(blob_type)

        elif issubclass(blob_type, BlobArray):
            return BlobArrayInterface(blob_type)

        elif blob_type.is_plain():
            return BlobPlainInterface(blob_type)

        else:
            return BlobComplexInterface(blob_type)

    @property
    def header_code(self):
        return self._header_code

    @property
    def source_code(self):
        return self._source_code

    def __init__(
            self,
            device=None,
            required_constant_blob_types=None,
            required_global_blob_types=None,
            header_header='',
            header_footer=''
    ):

        if device is None:
            device = opencl.create_some_context(False).get_info(opencl.context_info.DEVICES)[0]

        if required_constant_blob_types is None:
            required_constant_blob_types = []

        if required_global_blob_types is None:
            required_global_blob_types = []

        self.type_definitions = []
        self.function_definitions = []
        self.function_declarations = []

        for address_space_qualifier, required_blob_types in [
            ('constant', required_constant_blob_types),
            ('global', required_global_blob_types)
        ]:

            # check dependencies
            blob_types = []
            for blob_type in required_blob_types:
                blob_types.extend(blob_type.get_dependencies(recursive=True))
                blob_types.append(blob_type)

            # generate header
            generated_types = []
            for blob_type in blob_types:

                # ignore duplicates
                if blob_type in generated_types:
                    continue

                else:
                    generated_types.append(blob_type)

                # get interface
                blob_type_interface = BlobLib.get_interface(blob_type)

                # add type definition
                if implode_float_n:
                    type_definition = implode_floatn(
                        blob_type_interface.get_type(address_space_qualifier=address_space_qualifier))

                else:
                    type_definition = (blob_type_interface.get_type(address_space_qualifier=address_space_qualifier))

                self.type_definitions.append(type_definition)

                # add function definitions and declarations
                function_definitions, function_declarations = blob_type_interface.get_functions(
                    address_space_qualifier=address_space_qualifier
                )
                self.function_definitions.extend(function_definitions)
                self.function_declarations.extend(function_declarations)

        self._header_code = '\n\n'.join([
            '/* header generated by %s */' % __file__,
            header_header,
            '\n\n'.join(map(str.strip, self.type_definitions)),
            '\n\n'.join(map(str.strip, self.function_definitions)),
            header_footer
        ])

        self._source_code = '\n\n'.join([
            '/* source generated by %s */' % __file__,
            '\n\n'.join(map(str.strip, self.function_declarations))
        ])
