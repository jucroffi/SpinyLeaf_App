"""honeybee-doe2 translation commands."""
import sys
import json
import logging
import click

from ladybug.commandutil import process_content_to_output
from honeybee.typing import clean_doe2_string
from honeybee.model import Model
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from honeybee_energy.schedule.dictutil import dict_to_schedule

from honeybee_doe2.config import RES_CHARS
from honeybee_doe2.util import header_comment_minor
from honeybee_doe2.schedule import extract_all_schedule_ruleset_from_inp_file
from honeybee_doe2.simulation import SimulationPar

_logger = logging.getLogger(__name__)


@click.group(help='Commands for translating Honeybee Model to DOE-2 formats.')
def translate():
    pass


@translate.command('model-to-inp')
@click.argument('model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--sim-par-json', '-sp', help='Full path to a honeybee-doe2 SimulationPar '
    'JSON that describes all of the settings for the simulation. If unspecified, '
    'default parameters will be generated.', default=None, show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--hvac-mapping', '-hm', help='Text to indicate how HVAC systems should be '
    'assigned to the exported model. Story will assign one HVAC system for each '
    'distinct level polygon, Model will use only one HVAC system for the whole model '
    'and AssignedHVAC will follow how the HVAC systems have been assigned to the'
    'Rooms.properties.energy.hvac. Choose from: Room, Story, Model, AssignedHVAC',
    default='Story', show_default=True, type=str)
@click.option(
    '--include-interior-walls/--exclude-interior-walls', ' /-xw', help='Flag to note '
    'whether interior walls should be excluded from the export.',
    default=True, show_default=True)
@click.option(
    '--include-interior-ceilings/--exclude-interior-ceilings', ' /-xc', help='Flag to '
    'note whether interior ceilings should be excluded from the export.',
    default=True, show_default=True)
@click.option(
    '--equest-version', '-eq', help='Optional text string to denote the version '
    'of eQuest for which the INP definition will be generated. If unspecified '
    'or unrecognized, the latest version of eQuest will be used.',
    default='3.65', show_default=True, type=str)
@click.option(
    '--output-file', '-o', help='Optional INP file path to output the INP string '
    'of the translation. By default this will be printed out to stdout.',
    type=click.File('w'), default='-', show_default=True)
def model_to_inp_cli(
    model_file, sim_par_json, hvac_mapping, include_interior_walls,
    include_interior_ceilings, equest_version, output_file
):
    """Translate a Honeybee Model to an INP file.

    \b
    Args:
        model_file: Full path to a Honeybee Model file (HBJSON or HBpkl).
    """
    try:
        exclude_interior_walls = not include_interior_walls
        exclude_interior_ceilings = not include_interior_ceilings
        model_to_inp(
            model_file, sim_par_json, hvac_mapping,
            exclude_interior_walls, exclude_interior_ceilings,
            equest_version, output_file)
    except Exception as e:
        _logger.exception(f'Model translation failed:\n{e}')
        sys.exit(1)
    else:
        sys.exit(0)


def model_to_inp(
        model_file, sim_par_json=None, hvac_mapping='Story',
        exclude_interior_walls=False, exclude_interior_ceilings=False,
        equest_version='3.65', output_file=None,
        include_interior_walls=True, include_interior_ceilings=True):
    """Translate a Honeybee Model to an INP file.

    Args:
        model_file: Full path to a Honeybee Model file (HBJSON or HBpkl).
        sim_par_json: Full path to a honeybee-doe2 SimulationPar JSON that
            describes all of the settings for the simulation. If None,
            default parameters will be generated. (Default: None).
        hvac_mapping: Text to indicate how HVAC systems should be assigned to
            the exported model. Story will assign one HVAC system for each distinct
            level polygon, Model will use only one HVAC system for the whole model
            and AssignedHVAC will follow how the HVAC systems have been assigned
            to the Rooms.properties.energy.hvac. Choose from the following.

            * Room
            * Story
            * Model
            * AssignedHVAC

        exclude_interior_walls: Boolean to note whether interior walls should
            be excluded from the export. (Default: False).
        exclude_interior_ceilings: Boolean to note whether interior ceilings
            should be excluded from the export. (Default: False).
        equest_version: Optional text string to denote the version of eQuest for
            which the INP definition will be generated. If unspecified or
            unrecognized, the latest version of eQuest will be used. (Default: False).
        output_file: Optional INP file path to output the INP string of the
            translation. If None, the string will be returned from this function.
    """
    # load simulation parameters if specified
    sim_par = None
    if sim_par_json is not None:
        with open(sim_par_json) as json_file:
            data = json.load(json_file)
        sim_par = SimulationPar.from_dict(data)

    # re-serialize the Model to Python
    model = Model.from_file(model_file)

    # create the strings for the model
    inp_str = model.to.inp(
        model, sim_par, hvac_mapping,
        exclude_interior_walls, exclude_interior_ceilings, equest_version)

    # write out the INP file
    return process_content_to_output(inp_str, output_file)


@translate.command('schedules-to-inp')
@click.argument('schedule-json', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option('--output-file', '-f', help='Optional INP file to output the INP '
              'string of the translation. By default this will be printed out to stdout',
              type=click.File('w'), default='-', show_default=True)
def schedule_to_inp_cli(schedule_json, output_file):
    """Translate a Schedule JSON file to an INP.

    \b
    Args:
        schedule_json: Full path to a Schedule JSON file. This file should
            either be an array of non-abridged Schedules or a dictionary where
            the values are non-abridged Schedules.
    """
    try:
        schedule_to_inp(schedule_json, output_file)
    except Exception as e:
        _logger.exception('Schedule translation failed.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


def schedule_to_inp(schedule_json, output_file=None):
    """Translate a Schedule JSON file to an INP.

    Args:
        schedule_json: Full path to a Schedule JSON file. This file should
            either be an array of non-abridged Schedules or a dictionary where
            the values are non-abridged Schedules.
        output_file: Optional INP file path to output the INP string of the
            translation. If None, the string will be returned from this function.
    """
    # re-serialize the Schedule to Python
    with open(schedule_json) as json_file:
        data = json.load(json_file)
    sch_list = data.values() if isinstance(data, dict) else data
    sch_objs = [dict_to_schedule(sch) for sch in sch_list]
    type_objs = set()
    for sch in sch_objs:
        type_objs.add(sch.schedule_type_limit)

    # create the INP strings
    all_day_scheds, all_week_scheds, all_year_scheds = [], [], []
    used_day_sched_ids, used_day_count = {}, 1
    all_scheds = sch_objs
    for sched in all_scheds:
        if isinstance(sched, ScheduleRuleset):
            year_schedule, week_schedules = sched.to_inp()
            # check that day schedules aren't referenced by other model schedules
            day_scheds = []
            for day in sched.day_schedules:
                if day.identifier not in used_day_sched_ids:
                    day_scheds.append(day.to_inp(sched.schedule_type_limit))
                    used_day_sched_ids[day.identifier] = day
                elif day != used_day_sched_ids[day.identifier]:
                    new_day = day.duplicate()
                    new_day.identifier = 'Schedule Day {}'.format(used_day_count)
                    day_scheds.append(new_day.to_inp(sched.schedule_type_limit))
                    for i, week_sch in enumerate(week_schedules):
                        old_day_id = clean_doe2_string(day.identifier, RES_CHARS)
                        new_day_id = clean_doe2_string(new_day.identifier, RES_CHARS)
                        week_schedules[i] = week_sch.replace(old_day_id, new_day_id)
                    used_day_count += 1
            all_day_scheds.extend(day_scheds)
            all_week_scheds.extend(week_schedules)
            all_year_scheds.append(year_schedule)
        else:  # ScheduleFixedInterval
            year_schedule, week_schedules, year_schedule = sched.to_inp()
            all_day_scheds.extend(day_scheds)
            all_week_scheds.extend(week_schedules)
            all_year_scheds.append(year_schedule)
    inp_str_list = ['INPUT ..\n\n']
    inp_str_list.append(header_comment_minor('Day Schedules'))
    inp_str_list.extend(all_day_scheds)
    inp_str_list.append(header_comment_minor('Week Schedules'))
    inp_str_list.extend(all_week_scheds)
    inp_str_list.append(header_comment_minor('Annual Schedules'))
    inp_str_list.extend(all_year_scheds)
    inp_str_list.append('END ..\nCOMPUTE ..\nSTOP ..\n')
    inp_str = '\n'.join(inp_str_list)

    # write out the INP file
    return process_content_to_output(inp_str, output_file)


@translate.command('schedules-from-inp')
@click.argument('schedule-inp', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--dictionary/--array', ' /-a', help='Flag to note whether a the output JSON '
    'should be an array of schedule objects or whether it should be a dictionary '
    'where each key is the identifier of the schedule and each value is the '
    'schedule object. The dictionary format is the one used by honeybee-standards '
    'and is recommended when writing INP schedules into the user standards library.',
    default=True, show_default=True)
@click.option(
    '--output-file', '-f', help='Optional JSON file to output the JSON string of '
    'the translation. By default this will be printed out to stdout',
    type=click.File('w'), default='-', show_default=True)
def schedule_from_inp_cli(schedule_inp, dictionary, output_file):
    """Translate a schedule INP file to a honeybee JSON as an array of schedules.

    \b
    Args:
        schedule_inp: Full path to a Schedule INP file.
    """
    try:
        array = not dictionary
        schedule_from_inp(schedule_inp, array, output_file)
    except Exception as e:
        _logger.exception('Schedule translation failed.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


def schedule_from_inp(schedule_inp, array=False, output_file=None, dictionary=True):
    """Translate a schedule INP file to a honeybee JSON as an array of schedules.

    Args:
        schedule_inp: Full path to a Schedule INP file.
        array: Boolean to note whether a the output JSON should be an array of
            schedule objects or whether it should be a dictionary where each key
            is the identifier of the schedule and each value is the schedule object.
            The dictionary format is the one used by honeybee-standards and is
            recommended when writing INP schedules into the user standards
            library. (Default: False).
        output_file: Optional JSON file path to output the JSON string of the
            translation. If None, the string will be returned from this function.
    """
    # re-serialize the schedules to Python
    schedules = extract_all_schedule_ruleset_from_inp_file(schedule_inp)
    # create the honeybee dictionaries
    if array:
        hb_objs = [sch.to_dict() for sch in schedules]
    else:
        hb_objs = {sch.identifier: sch.to_dict() for sch in schedules}
    # write out the JSON file
    content_str = json.dumps(hb_objs)
    return process_content_to_output(content_str, output_file)
