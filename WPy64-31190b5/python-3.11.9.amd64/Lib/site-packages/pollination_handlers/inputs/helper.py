import os
import tempfile
import uuid

_BOOL_STRINGS = ('True', 'False')


def bool_option_to_str(value, acceptable_strings, input_name=''):
    """Convert a boolean option to a string given a list of acceptable values."""
    acceptable = acceptable_strings + _BOOL_STRINGS
    if isinstance(value, str):
        assert value in acceptable, '{} value "{}" is not acceptable. ' \
            'Must be one of the following: {}'.format(input_name, value, acceptable)
        if value in _BOOL_STRINGS:
            result = acceptable[0] if value == 'True' else acceptable[1]
        else:
            result = value
    elif isinstance(value, bool):
        result = acceptable[0] if value else acceptable[1]
    else:
        raise ValueError(
            '{} input should be a string or a boolean. '
            'Got {}.'.format(input_name, type(value))
        )
    return result


def get_tempfile(extension, file_name=None):
    """Get full path to a temporary file with extension."""
    file_name = str(uuid.uuid4())[:6] if file_name is None \
        or file_name == '-' else file_name
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, '%s.%s' % (file_name, extension))
    return file_path


def get_tempfolder(folder_name=None):
    """Get full path to a temporary folder with extension."""
    folder_name = str(uuid.uuid4())[:6] if folder_name is None \
        or folder_name == '-' else folder_name
    temp_dir = tempfile.gettempdir()
    folder_path = os.path.join(temp_dir, folder_name)
    os.mkdir(folder_path)
    return folder_path


def write_values_to_csv(file_path, values):
    """Write a list of values to a CSV."""
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(values))
    return file_path


def write_sch_values_to_csv(file_path, values):
    """Write a list of fractional values to a discrete 0/1 CSV."""
    discrete_vals = []
    for v in values:
        dv = '1' if v >= 0.1 else '0'
        discrete_vals.append(dv)
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(discrete_vals))
    return file_path
