"""Collection of auxilary functions for radiance commands and options."""
import re
import collections
import os
import sys

if (sys.version_info < (3, 0)):
    import urllib2
    readmode = 'rb'
    writemode = 'wb'
else:
    import urllib.request
    readmode = 'r'
    writemode = 'w'


_rad_opt_pattern = r'-[a-zA-Z]+'
_rad_opt_compiled_pattern = re.compile(_rad_opt_pattern)


def parse_radiance_options(string):
    """Parse a radiance options string (e.g. '-ab 4 -ad 256').

    The string should start with a '-' otherwise it will be trimmed to the first '-' in
    string.
    """
    try:
        index = string.index('-')
    except ValueError:
        if not ' '.join(string.split()).replace('"', '').replace("'", '').strip():
            return {}
        raise ValueError(
            'Invalid Radiance options string input. Failed to find - in input string.'
        )

    sub_string = ' '.join(string[index:].split())
    value = re.split(_rad_opt_compiled_pattern, sub_string)[1:]
    key = re.findall(_rad_opt_pattern, sub_string)

    options = collections.OrderedDict()
    for k, v in zip(key, value):
        values = v.split()
        count = len(values)
        if count == 0:
            values = ''
        elif count == 1:
            values = values[0]
        options[k[1:]] = values

    return options


def nukedir(target_dir, rmdir=False):
    """Delete all the files inside target_dir.

    Usage:
        nukedir("c:/ladybug/libs", True)
    """
    d = os.path.normpath(target_dir)

    if not os.path.isdir(d):
        return

    files = os.listdir(d)

    for f in files:
        if f == '.' or f == '..':
            continue
        path = os.path.join(d, f)

        if os.path.isdir(path):
            nukedir(path)
        else:
            try:
                os.remove(path)
            except Exception:
                print("Failed to remove %s" % path)

    if rmdir:
        try:
            os.rmdir(d)
        except Exception:
            print("Failed to remove %s" % d)


def preparedir(target_dir, remove_content=True):
    """Prepare a folder for analysis.

    This method creates the folder if it is not created, and removes the file in
    the folder if the folder already existed.
    """
    if os.path.isdir(target_dir):
        if remove_content:
            nukedir(target_dir, False)
        return True
    else:
        try:
            os.makedirs(target_dir)
            return True
        except Exception as e:
            print("Failed to create folder: %s\n%s" % (target_dir, e))
            return False


def write_to_file_by_name(folder, fname, data, mkdir=False):
    """Write a string of data to file by filename and folder.

    Args:
        folder: Target folder (e.g. c:/ladybug).
        fname: File name (e.g. testPts.pts).
        data: Any data as string.
        mkdir: Set to True to create the directory if doesn't exist (Default: False).
    """
    if not os.path.isdir(folder):
        if mkdir:
            preparedir(folder)
        else:
            created = preparedir(folder, False)
            if not created:
                raise ValueError("Failed to find %s." % folder)

    file_path = os.path.join(folder, fname)

    with open(file_path, writemode) as outf:
        try:
            outf.write(str(data))
            return file_path
        except Exception as e:
            raise IOError("Failed to write %s to file:\n\t%s" % (fname, str(e)))
