
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
                result['%s_%s' % (camel_case_to_underscore(key), subkey)] = value
        elif isinstance(value, list):
            for index, value in enumerate(value):
                result['%s_%d' % (camel_case_to_underscore(key), index)] = value
        else:
            result[camel_case_to_underscore(key)] = value
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


def get_blob_index(dtype, name):
    for index, field_name in enumerate(dtype.names):
        if field_name == name:
            return index


def dtype_to_lines(dtype):
    return str(dtype).replace('), (', ')\n(').split('\n')