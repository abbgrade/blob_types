"""
... is a submodule of [blob_types](__init__.html).
It contains helper functions.
"""
import os

implode_float_n = False

def camel_case_to_underscore(name):
    """Convert a CamelCaseString into an underscore_string."""

    new_name = []
    for char in name:
        if char.isupper() and len(new_name) > 0:
            new_name.append('_' + char.lower())

        else:
            new_name.append(char.lower())

    return ''.join(new_name)

def underscore_to_camel_case(name):
    """Convert a underscore_string into a CamelCaseString."""

    name = ''.join(map(lambda part: part.capitalize(), name.split('_')))
    name = name[0].lower() + name[1:]
    return name

def flat_struct(struct):
    """Copy a struct, which contains all values but without nesting dicts of lists.

    ### Example:
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

        # trim key
        while key.endswith('_'):
            key = key[:-1]

        if isinstance(value, dict):
            for subkey, value in flat_struct(value).items():
                result['%s_%s' % (camel_case_to_underscore(key), subkey)] = value

        elif isinstance(value, list):
            for index, value in enumerate(value):
                result['%s_%d' % (camel_case_to_underscore(key), index)] = value

        else:
            result[camel_case_to_underscore(key)] = value

    return result


vector_fields = [
    ('x', 'y', 'z'),
    ('0', '1', '2')
]

def implode_floatn(struct_source):
    """Convert vector types."""
    lines = []
    for line in struct_source.split('\n'):
        parts = line.split()
        replaced = False
        if len(parts) == 2 and parts[0].startswith('float'):
            type_, field = parts
            previous_parts = lines[-1].split()
            if len(previous_parts) == 2:
                previous_type, previous_field = previous_parts

                for vector_field in vector_fields:
                    if not replaced:
                        for previous_index, char in enumerate(vector_field):
                            if not replaced:
                                index = str(previous_index + 1)
                                if previous_index < 2:
                                    previous_index = ''

                                else:
                                    previous_index = str(previous_index)

                                suffix = '_%s;' % char
                                new_type = 'float%s' % index

                                previous_prefix = previous_field[:-len(suffix)]
                                prefix = field[:-len(suffix)]

                                if field.endswith(suffix) and previous_prefix == prefix or (
                                    previous_index and previous_field[:-1] == prefix):
                                    new_line = line.replace('float', new_type)[:-len(suffix)] + ';'
                                    lines[-1] = new_line
                                    replaced = True

        if not replaced:
            lines.append(line)

    return '\n'.join(map(lambda line_: line_.replace('float3 ', 'float4 '), lines))

def get_blob_index(dtype, name):
    assert dtype
    assert dtype.names

    for index, field_name in enumerate(dtype.names):
        if field_name == name:
            return index

def dtype_to_lines(dtype):
    return str(dtype).replace('), (', ')\n(').split('\n')

def diff_buffer(a, b, dtype=None):
    result = ['len(a):%d == %d:len(b)' % (len(a), len(b))]

    assert a is not None
    assert b is not None

    length = max(len(a), len(b))
    diff_start = None

    for index in range(length):

        if diff_start and diff_start + 5 <= index:
            break

        a_field, b_field = a[index], b[index]
        if a_field != b_field:

            # determine field were the difference is
            diff_field = None
            if dtype:
                previous_field = None
                for field in dtype.names:
                    field_dtype, field_offset = dtype.fields[field]
                    if index < field_offset:
                        diff_field = previous_field
                        break
                    previous_field = field

            result.append('%d %s: %d != %d' % (index, diff_field, ord(a_field), ord(b_field)))

            if diff_start is None:
                diff_start = index

    return '\n'.join(result)

def diff_dtype(a, b):
    result = ['diff(%s, %s)' % (a.name, b.name)]

    assert a is not None
    assert b is not None
    assert a.names is not None, a
    assert b.names is not None, b

    length = max(len(a), len(b))
    diff_start = None

    for index in range(length):

        if diff_start and diff_start + 5 <= index:
            break

        a_field, b_field = None, None

        try:
            a_field = a.names[index]
            b_field = b.names[index]

        except IndexError:

            if len(a) <= index:
                result.append('%d: %s missing on first dtype' % (index, b_field))

                if diff_start is None:
                    diff_start = index

            elif len(b) <= index:
                result.append('%d: %s missing on second dtype' % (index, a_field))

                if diff_start is None:
                    diff_start = index

        else:

            if a_field != b_field:
                result.append('%d: %s != %s' % (index, a_field, b_field))

                if diff_start is None:
                    diff_start = index

            elif a_field.isdigit():
                if diff_start is None:
                    result.append('%d: %s' % (index - 1, a.names[index - 1][0]))

                result.append('%d: %s is invalid on first dtype' % (index, a_field))

                if diff_start is None:
                    diff_start = index

            elif b_field.isdigit():
                if diff_start is None:
                    result.append('%d: %s' % (index - 1, b.names[index - 1][0]))
                result.append('%d: %s is invalid on second dtype' % (index, b_field))

                if diff_start is None:
                    diff_start = index

    return '\n'.join(result)

def diff_blob(a, b):
    return a.__diff__(b)

def src_path(root, path):
    if path.startswith('.'):
        path_list = os.path.dirname(root).split(os.sep)
        path_list.extend(path.split('/'))
        return os.sep.join(path_list)

    else:
        return path
