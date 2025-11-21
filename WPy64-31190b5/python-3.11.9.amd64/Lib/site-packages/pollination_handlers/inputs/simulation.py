"""Handlers for honeybee simulation parameters."""
import os
import json
try:
    from collections.abc import Iterable  # python < 3.7
except ImportError:
    from collections import Iterable  # python >= 3.8

from ladybug.futil import copy_file_tree, nukedir
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.measure import Measure

from .helper import get_tempfile, get_tempfolder


def energy_sim_par_to_json(sim_par_obj):
    """Translate a honeybee-energy SimulationParameter to a JSON file.

        Args:
            sim_par_obj: Either a honeybee-energy SimulationParameter or the path
                to the JSON file. In case the sim_par_obj is a path, it will be
                returned as is. For an object it will be saved to a HBJSON file
                in a temp folder.

        Returns:
            str -- Path to HBJSON file.
    """
    if isinstance(sim_par_obj, str):
        if not os.path.isfile(sim_par_obj):
            raise ValueError('Invalid file path: %s' % sim_par_obj)
        sp_file = sim_par_obj
    elif isinstance(sim_par_obj, SimulationParameter):
        sp_file = get_tempfile('json')
        obj_dict = sim_par_obj.to_dict()
        # write the dictionary into a file
        try:
            with open(sp_file, 'w') as fp:
                json.dump(obj_dict, fp)
        except UnicodeDecodeError:  # non-unicode character in the dictionary
            with open(sp_file, 'w') as fp:
                json.dump(obj_dict, fp, ensure_ascii=False)
    else:
        raise ValueError(
            'Simulation Parameter input should be a string or an object. '
            'Not {}.'.format(type(sim_par_obj))
        )
    return sp_file


def measures_to_folder(measures_obj):
    """Translate a list of honeybee-energy Measures to a folder.

        Args:
            measures_obj: Either a list of honeybee-energy Measure objects or
                the path to a folder that contains the measures and a .osw with
                the measure arguments. In case the measures_obj is a folder, it will be
                returned as is. For a list of objects it will be saved to an OSW file
                in a temp folder.

        Returns:
            str -- Path to a measures folder.
    """
    if isinstance(measures_obj, Measure):
        measures_obj = [measures_obj]
    if measures_obj is None:
        return ''
    elif isinstance(measures_obj, str):
        if measures_obj == '':
            return ''
        if not os.path.isdir(measures_obj):
            raise ValueError('Invalid folder path: %s' % measures_obj)
        osw_found = False
        for f in os.listdir(measures_obj):
            f_lower = f.lower()
            if f_lower.endswith('.osw'):
                osw_found = True
            elif f_lower == 'measure.rb':
                raise ValueError(
                    'Measure folder must contain the constituent measures '
                    'in sub-directories.')
        if not osw_found:
            raise ValueError('No .osw file was found in: %s' % measures_obj)
        mea_folder = measures_obj
    elif isinstance(measures_obj, Iterable):
        if len(measures_obj) == 0:
            return ''
        osw_dict = {}  # dictionary that will be turned into the OSW JSON
        osw_dict['steps'] = []
        mea_folder = get_tempfolder()  # will become the folder with all the measures
        # ensure measures are correctly ordered
        m_dict = {'ModelMeasure': [], 'EnergyPlusMeasure': [], 'ReportingMeasure': []}
        for measure in measures_obj:
            assert isinstance(measure, Measure), 'Expected honeybee-energy Measure. ' \
                'Got {}.'.format(type(measure))
            m_dict[measure.type].append(measure)
        sorted_measures = m_dict['ModelMeasure'] + m_dict['EnergyPlusMeasure'] + \
            m_dict['ReportingMeasure']
        # add the measures and the measure paths to the OSW
        for measure in sorted_measures:
            measure.validate()  # ensure that all required arguments have values
            osw_dict['steps'].append(measure.to_osw_dict())  # add measure to workflow
            dest_folder = os.path.join(mea_folder, os.path.basename(measure.folder))
            copy_file_tree(measure.folder, dest_folder)
            test_dir = os.path.join(dest_folder, 'tests')
            if os.path.isdir(test_dir):
                nukedir(test_dir, rmdir=True)
        # write the dictionary to a workflow.osw
        osw_json = os.path.join(mea_folder, 'workflow.osw')
        try:
            with open(osw_json, 'w') as fp:
                json.dump(osw_dict, fp, indent=4)
        except UnicodeDecodeError:  # non-unicode character in the dictionary
            with open(osw_json, 'w') as fp:
                json.dump(osw_dict, fp, indent=4, ensure_ascii=False)
    else:
        raise ValueError(
            'Measure input should be a list of Measure objects or a path to a folder. '
            'Not {}.'.format(type(measures_obj))
        )
    return mea_folder


def list_to_additional_strings(additional_strings):
    """Translate a list of additional strings into a single string.

        Args:
            additional_strings: Either a single string or a list of strings to be
                joined into one.

        Returns:
            str -- A single IDF string.
    """
    if additional_strings is None or additional_strings == '':
        return ''
    elif isinstance(additional_strings, str):
        return additional_strings
    elif isinstance(additional_strings, Iterable):
        return '\n'.join(list(additional_strings))
    else:
        raise ValueError(
            'Additional strings input should be a list or a single string. '
            'Not {}.'.format(type(additional_strings))
        )


def list_to_additional_idf(additional_strings):
    """Translate a list of additional strings into a single IDF file.

        Args:
            additional_strings: Either a single string or a list of strings to be
                written into an IDF file.

        Returns:
            str -- Path to an IDF file.
    """
    base_str = list_to_additional_strings(additional_strings)
    if base_str != '':
        add_idf = get_tempfile('idf')
        with open(add_idf, 'w') as fp:
            fp.write(base_str)
        return add_idf
    return ''


def viz_variables_to_string(viz_variables):
    """Translate a list of visualization variables into a single string.

        Args:
            viz_variables: Either a single string or a list of strings to be
                joined into one.

        Returns:
            str -- A single IDF string.
    """
    if viz_variables is None or viz_variables == '':
        return ''
    elif isinstance(viz_variables, str):
        if not viz_variables.startswith('-v') and not \
                viz_variables.startswith('--viz-variable'):
            viz_variables = '-v "{}"'.format(viz_variables)
        return viz_variables
    elif isinstance(viz_variables, Iterable):
        viz_variables = ['-v "{}"'.format(var) for var in viz_variables]
        return ' '.join(viz_variables)
    else:
        raise ValueError(
            'Visualization variables input should be a list or a single string. '
            'Not {}.'.format(type(viz_variables))
        )


def standard_to_str(standard_str):
    """Translate a different text inputs into an acceptable standard/vintage.

        Args:
            standard_str: A string referencing a standard or building vintage
                to be used.

        Returns:
            str -- A string that correctly references the standard in honeybee-schema.
    """
    EFF_STANDARDS = {
        'DOE_Ref_Pre_1980': 'DOE_Ref_Pre_1980',
        'DOE_Ref_1980_2004': 'DOE_Ref_1980_2004',
        'ASHRAE_2004': 'ASHRAE_2004',
        'ASHRAE_2007': 'ASHRAE_2007',
        'ASHRAE_2010': 'ASHRAE_2010',
        'ASHRAE_2013': 'ASHRAE_2013',
        'ASHRAE_2016': 'ASHRAE_2016',
        'ASHRAE_2019': 'ASHRAE_2019',
        'pre_1980': 'DOE_Ref_Pre_1980',
        '1980_2004': 'DOE_Ref_1980_2004',
        '2004': 'ASHRAE_2004',
        '2007': 'ASHRAE_2007',
        '2010': 'ASHRAE_2010',
        '2013': 'ASHRAE_2013',
        '2016': 'ASHRAE_2016',
        '2019': 'ASHRAE_2019'
    }
    if standard_str is not None and standard_str != '':
        return EFF_STANDARDS[standard_str]
