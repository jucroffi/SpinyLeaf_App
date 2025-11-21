"""Handlers for the summary .json objects and folder contents."""
import os
import json


def json_properties_from_path(summary_json):
    """Read the properties from a summary json file."""
    if not os.path.isfile(summary_json):
        raise ValueError('Invalid file path: %s' % summary_json)
    with open(summary_json) as json_file:
        data = json.load(json_file)
    results = []
    for prop, value in data.items():
        results.append('{}: {}'.format(prop, value))
    return results


def contents_from_folder(output_folder):
    """Read the contents of all files in a folder."""
    if not os.path.isdir(output_folder):
        raise ValueError('Invalid folder path: %s' % output_folder)
    results = []
    for f in os.listdir(output_folder):
        results.append(os.path.join(output_folder, f))
    return results
