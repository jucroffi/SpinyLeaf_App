"""Collection of methods for type input checking."""
import re
import os
import math

try:
    INFPOS = math.inf
    INFNEG = -1 * math.inf
except AttributeError:
    # python 2
    INFPOS = float('inf')
    INFNEG = float('-inf')


def valid_string(value, input_name=''):
    """Get a valid string for both Radiance and EnergyPlus.

    This is used for honeybee face and honeybee zone names.
    """
    try:
        val = re.sub(r'[^.A-Za-z0-9_-]', '', value)
    except TypeError:
        raise TypeError('Input {} must be a text string. Got {}: {}.'.format(
            input_name, type(value), value))
    assert len(val) > 0, 'Input {} "{}" contains no valid characters.'.format(
        input_name, value)
    assert len(val) <= 100, 'Input {} "{}" must be less than 100 characters.'.format(
        input_name, value)
    return val


def valid_rad_string(value, input_name=''):
    """Get a valid string for Radiance that can be used for rad material names, etc.

    This includes stripping out illegal characters and white spaces.
    """
    try:
        val = re.sub(r'[^.A-Za-z0-9_-]', '', value)
    except TypeError:
        raise TypeError('Input {} must be a text string. Got {}: {}.'.format(
            input_name, type(value), value))
    assert len(val) > 0, 'Input {} "{}" contains no valid characters.'.format(
        input_name, value)
    return val


def valid_ep_string(value, input_name=''):
    """Get valid string for EnergyPlus that can be used for energy material names, etc.

    This includes stripping out all illegal characters, removing trailing white spaces,
    and ensuring the name is not longer than 100 characters.
    """
    try:
        val = ''.join(i for i in value if ord(i) < 128)  # strip out non-ascii
        val = re.sub(r'[,;!\n\t]', '', val)  # strip out E+ special characters
    except TypeError:
        raise TypeError('Input {} must be a text string. Got {}: {}.'.format(
            input_name, type(value), value))
    val = val.strip()
    assert len(val) > 0, 'Input {} "{}" contains no valid characters.'.format(
        input_name, value)
    assert len(val) <= 100, 'Input {} "{}" must be less than 100 characters.'.format(
        input_name, value)
    return val


def float_in_range(value, mi=INFNEG, ma=INFPOS, input_name=''):
    """Check a float value to be between minimum and maximum."""
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise TypeError('Input {} must be a number. Got {}: {}.'.format(
            input_name, type(value), value))
    assert mi <= number <= ma, 'Input number {} must be between {} and {}. ' \
        'Got {}'.format(input_name, mi, ma, value)
    return number


def int_in_range(value, mi=INFNEG, ma=INFPOS, input_name=''):
    """Check an integer value to be between minimum and maximum."""
    try:
        number = int(value)
    except ValueError:
        # try to convert to float and then digit if possible
        try:
            number = int(float(value))
        except (ValueError, TypeError):
            raise TypeError('Input {} must be an integer. Got {}: {}.'.format(
                input_name, type(value), value))
    except (ValueError, TypeError):
        raise TypeError('Input {} must be an integer. Got {}: {}.'.format(
            input_name, type(value), value))
    assert mi <= number <= ma, 'Input integer {} must be between {} and {}. ' \
        'Got {}.'.format(input_name, mi, ma, value)
    return number


def float_positive(value, input_name=''):
    """Check a float value to be positive."""
    return float_in_range(value, 0, INFPOS, input_name)


def int_positive(value, input_name=''):
    """Check if an integer value is positive."""
    return int_in_range(value, 0, INFPOS, input_name)


def tuple_with_length(value, length=3, item_type=float, input_name=''):
    """Try to create a tuple with a certain value."""
    try:
        value = tuple(item_type(v) for v in value)
    except (ValueError, TypeError):
        raise TypeError('Input {} must be a {}.'.format(
            input_name, item_type))
    assert len(value) == length, 'Input {} length must be {} not {}'.format(
        input_name, length, len(value))
    return value


def list_with_length(value, length=3, item_type=float, input_name=''):
    """Try to create a list with a certain value."""
    try:
        value = [item_type(v) for v in value]
    except (ValueError, TypeError):
        raise TypeError('Input {} must be a {}.'.format(
            input_name, item_type))
    assert len(value) == length, 'Input {} length must be {} not {}'.format(
        input_name, length, len(value))
    return value


wrapper = '"' if os.name == 'nt' else '\''
"""String wrapper."""


def normpath(value):
    """Normalize a path by eliminating double slashes, etc and put it in quotes if
    needed. Then, convert all the slashes are forwarded slashes to maintain compatibility
    across platforms"""

    value = os.path.normpath(value)

    # Needed For Windows.
    value = value.replace('\\', '/')
    if ' ' in value:
        value = '{0}{1}{0}'.format(wrapper, value)
    return value


def path_checker(file_path, extn_list=None, file_descr=''):
    """A utility method to check for input file path and normalize the path if present.
    If extension list and file_descr are provided, then do additional checks to
    ensure that the correct file type has been specified.

    Args:
        file_path: The path of the input. Will be normalized through 'typing'
        extn_list: List of extensions, including the leading dot e.g. ['.hdr','.pic']
            which if provided will be checked for before the path is set. If there
            is no match, a value error will be raised.
        file_descr: A phrase describing the file (e.g. 'Radiance HDR') that can be
            used to compose the error message in case the extension check fails.
    """

    if file_path is None:
        return None
    else:
        file_name, file_extn = os.path.splitext(file_path)

        if extn_list:
            assert type(extn_list) in (list, tuple), \
                'The input for extn_list(%s) should either be a list or a tuple that ' \
                'contains extensions (e.g. [".jpg",".bmp"]'

        if extn_list and file_extn.lower() not in extn_list:
            file_descr = 'for %s files' % file_descr if file_descr else ''
            err_msg = 'The provided input (%s) is invalid %s as only a file with the ' \
                      'following extensions is allowed: %s' % (file_path, file_descr,
                                                               ','.join(extn_list))
            raise ValueError(err_msg)
        return normpath(file_path)


def path_checker_multiple(file_paths, extn_list=None, file_descr='',
                          outputs_as_string=False):
    """A utility method to check for list (or tuple) of input file paths and
        return either a list of normalized paths or a single string containing
        space-separated normalized paths. If extension list and file_descr are
        provided, then do additional checks to ensure that the correct file type
        has been specified.

    Args:
        file_paths: List of input file paths. Will be normalized through 'typing'
        extn_list: List of extensions, including the leading dot e.g. ['.hdr','.pic']
            which if provided will be checked for before the path is set. If there
            is no match, a value error will be raised.
        file_descr: A phrase describing the file (e.g. 'Radiance HDR') that can be
            used to compose the error message in case the extension check fails.
        outputs_as_string: If set to True, the files will be returned as single
            string of normalized paths. Else a list of normalized paths will
            be returned.
    """

    if not file_paths:
        file_paths = []
    elif not isinstance(file_paths, (list, tuple)):
        file_paths = [file_paths]

    final_path_list = [path_checker(file_path,
                                    extn_list=extn_list, file_descr=file_descr)
                       for file_path in file_paths]

    if outputs_as_string:
        return ' '.join(final_path_list)
    else:
        return final_path_list
