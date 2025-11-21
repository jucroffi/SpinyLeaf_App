"""honeybee-doe2 program type translators."""
from __future__ import division

from .util import switch_statement_id
from .load import people_to_inp, lighting_to_inp, electric_equipment_to_inp, \
    infiltration_to_inp, setpoint_to_inp, ventilation_to_inp, \
    SPACE_KEYS, ZONE_KEYS, SCHEDULE_KEYS, MECH_AIRFLOW_KEYS
SCH_KEY_SET = set(SCHEDULE_KEYS)


def program_type_to_inp(program_type, switch_dict=None):
    """Translate a ProgramType into a dictionary used to write INP switch statements.

    Args:
        program_type: A honeybee-energy ProgramType definition.
        switch_dict: An optional dictionary with INP keywords as keys (such as
            PEOPLE-SCHEDULE or AREA/PERSON or LIGHTING-W/AREA). The values of the
            dictionary should be lists of switch statement text strings, such as
            'case "conf": #SI("Small Off Occ", "SPACE", "PEOPLE-SCHEDULE")'.
            Specifying an input dictionary here can be used to build up switch
            statements for all program types across a model.

    Returns:
        An dictionary with INP keywords as keys (such as PEOPLE-SCHEDULE or AREA/PERSON).
        The values of the dictionary are lists of switch statement text strings,
        such as 'case "conf": #SI("Small Off Occ", "SPACE", "PEOPLE-SCHEDULE")'.
    """
    # set up the switch statement dictionary to be filled
    switch_dict = switch_dict if switch_dict is not None else {}
    prog_uid = switch_statement_id(program_type.identifier)
    base_switch = 'case "{}": '.format(prog_uid)

    def _format_schedule(sch_key, sch_uid, obj_type='SPACE'):
        """Format schedules in the way they are written into switch statements."""
        return '{}#SI({}, "{}", "{}")'.format(base_switch, sch_uid, obj_type, sch_key)
    
    def _add_to_switch_dict(keyword, value):
        """Add a key: value pair to the switch dictionary with a check."""
        try:
            switch_dict[keyword].append(value)
        except KeyError:  # the first time this key was encountered in the dict
            switch_dict[keyword] = [value]

    # write the people into the dictionary
    ppl_kwd, ppl_val = people_to_inp(program_type.people)
    for key, val in zip(ppl_kwd, ppl_val):
        if key in SCH_KEY_SET:
            _add_to_switch_dict(key, _format_schedule(key, val, 'SPACE'))
        else:
            _add_to_switch_dict(key, '{}{}'.format(base_switch, val))


    # write the lighting into the dictionary
    lgt_kwd, lgt_val = lighting_to_inp(program_type.lighting)
    for key, val in zip(lgt_kwd, lgt_val):
        if key == 'LIGHTING-SCHEDULE':
            key = 'LIGHTING-SCHEDUL'  # there's a typo in DOE-2 that was never fixed
            _add_to_switch_dict(key, _format_schedule(key, val, 'SPACE'))
        else:
            _add_to_switch_dict(key, '{}{}'.format(base_switch, val))

    # write the equipment into the dictionary
    eq_kwd, eq_val = electric_equipment_to_inp(program_type.electric_equipment)
    for key, val in zip(eq_kwd, eq_val):
        if key in SCH_KEY_SET:
            _add_to_switch_dict(key, _format_schedule(key, val, 'SPACE'))
        else:
            _add_to_switch_dict(key, '{}{}'.format(base_switch, val))

    # write the infiltration into the dictionary
    inf_kwd, inf_val = infiltration_to_inp(program_type.infiltration)
    for key, val in zip(inf_kwd, inf_val):
        if key in SCH_KEY_SET:
            _add_to_switch_dict(key, _format_schedule(key, val, 'SPACE'))
        elif key == 'INF-METHOD':
            continue  # DOE-2 does not like when we define this key
        else:
            _add_to_switch_dict(key, '{}{}'.format(base_switch, val))

    # write the setpoint into the dictionary
    stp_kwd, stp_val = setpoint_to_inp(program_type.setpoint)
    for key, val in zip(stp_kwd, stp_val):
        if key in SCH_KEY_SET:
            _add_to_switch_dict(key, _format_schedule(key, val, 'ZONE'))
        else:
            _add_to_switch_dict(key, '{}{}'.format(base_switch, val))

    # write the ventilation into the dictionary
    vt_kwd, vt_val = ventilation_to_inp(program_type.ventilation)
    for key, val in zip(vt_kwd, vt_val):
        if key in SCH_KEY_SET:
            _add_to_switch_dict(key, _format_schedule(key, val, 'ZONE'))
        else:
            _add_to_switch_dict(key, '{}{}'.format(base_switch, val))

    # if the user_data of the ProgramType has Mech AirFlow keys, add them
    if program_type.user_data is not None:
        for air_key in MECH_AIRFLOW_KEYS:
            if air_key in program_type.user_data:
                val = program_type.user_data[air_key]
                _add_to_switch_dict(air_key, '{}{}'.format(base_switch, val))

    return switch_dict


def switch_dict_to_space_inp(switch_dict):
    """Translate a switch statement dictionary into INP strings for the SPACE.

    Args:
        switch_dict: An dictionary with INP keywords as keys (such as
            PEOPLE-SCHEDULE or AREA/PERSON or LIGHTING-W/AREA). The values of the
            dictionary should be lists of switch statement text strings, such as
            'case "conf": #SI("Small Off Occ", "SPACE", "PEOPLE-SCHEDULE")'.

    Returns:
        A text string to be written into an INP file. This should go at the top
        of the description of Floors / Spaces.
    """
    # loop through the space keys and build a list of all switch statement keys
    all_switch_strs = []
    for s_key in SPACE_KEYS:
        try:
            if len(s_key) > 16:  # switch statements limit characters to 16
                s_key = s_key[:16]
            switch_progs = switch_dict[s_key]
            switch_strs = ['SET-DEFAULT FOR SPACE']
            switch_strs.append('   {} ='.format(s_key))
            switch_strs.append('{switch(#L("C-ACTIVITY-DESC"))')
            switch_strs.extend(switch_progs)
            switch_strs.append('default: no_default')
            switch_strs.append('endswitch}')
            switch_strs.append('..\n')
            all_switch_strs.append('\n'.join(switch_strs))
        except KeyError:
            pass  # none of the programs types have this space key
    # add something to set the INF-METHOD for all spaces
    inf_str = 'SET-DEFAULT FOR SPACE\n' \
        '   INF-METHOD = AIR-CHANGE\n' \
        '..\n'
    all_switch_strs.append(inf_str)
    return '\n'.join(all_switch_strs)


def switch_dict_to_zone_inp(switch_dict):
    """Translate a switch statement dictionary into INP strings for the ZONE.

    Args:
        switch_dict: An dictionary with INP keywords as keys (such as
            HEAT-TEMP-SCH or DESIGN-HEAT-T or OUTSIDE-AIR-FLOW). The values of the
            dictionary should be lists of switch statement text strings, such as
            'case "conf": #SI("Small Off HtgStp", "SPACE", "HEAT-TEMP-SCH")'.

    Returns:
        A text string to be written into an INP file. This should go at the top
        of the  description of HVAC Systems / Zones.
    """
    # loop through the space keys and build a list of all switch statement keys
    all_switch_strs = []
    for s_key in ZONE_KEYS:
        try:
            switch_progs = switch_dict[s_key]
            switch_strs = ['SET-DEFAULT FOR ZONE', '   TYPE = CONDITIONED']
            switch_strs.append('   {} ='.format(s_key))
            switch_strs.append('{switch(#LR("SPACE", "C-ACTIVITY-DESC"))')
            switch_strs.extend(switch_progs)
            switch_strs.append('default: no_default')
            switch_strs.append('endswitch}')
            switch_strs.append('..\n')
            all_switch_strs.append('\n'.join(switch_strs))
        except KeyError:
            pass  # none of the programs types have this space key
    return '\n'.join(all_switch_strs)
