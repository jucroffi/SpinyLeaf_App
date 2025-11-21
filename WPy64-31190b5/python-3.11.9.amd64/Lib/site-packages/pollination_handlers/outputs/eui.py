"""Handlers for the eui.json object output by annual energy use runs."""
import os
import json


def eui_json_from_path(eui_json):
    """Read the energy use values from the eui.json file."""
    if not os.path.isfile(eui_json):
        raise ValueError('Invalid file path: %s' % eui_json)
    with open(eui_json) as json_file:
        data = json.load(json_file)
    results = ['total: {}'.format(data['eui'])]
    for end_use, end_usage in data['end_uses'].items():
        results.append('{}: {}'.format(end_use, end_usage))
    return results
