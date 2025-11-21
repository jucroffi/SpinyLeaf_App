# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.radiant_system_controls.rb
"""
from __future__ import division

from ladybug.datatype.temperature import Temperature

from honeybee_openstudio.openstudio import openstudio, openstudio_model
from .utilities import ems_friendly_name
from .schedule import create_constant_schedule_ruleset, create_schedule_type_limits
from .thermal_zone import thermal_zone_get_occupancy_schedule

TEMPERATURE = Temperature()


def model_add_radiant_proportional_controls(
        model, zone, radiant_loop,
        radiant_temperature_control_type='SurfaceFaceTemperature',
        use_zone_occupancy_for_control=True, occupied_percentage_threshold=0.10,
        model_occ_hr_start=6.0, model_occ_hr_end=18.0,
        proportional_gain=0.3, switch_over_time=24.0):
    """Implement a proportional control for a single thermal zone with a radiant system.

    Args:
        model: Openstudio Model.
        zone: [OpenStudio::Model::ThermalZone>] zone to add radiant controls.
        radiant_loop: [OpenStudio::Model::ZoneHVACLowTempRadiantVarFlow>] radiant
            loop in thermal zone.
        radiant_temperature_control_type: [String] determines the controlled
            temperature for the radiant system. Options include the following:

            * SurfaceFaceTemperature
            * SurfaceInteriorTemperature

        use_zone_occupancy_for_control: [Boolean] Set to true if radiant system
            is to use specific zone occupancy objects for CBE control strategy.
            If false, then it will use values in model_occ_hr_start and model_occ_hr_end
            for all radiant zones. default to true.
        occupied_percentage_threshold: [Double] the minimum fraction (0 to 1) that
            counts as occupied. If this parameter is set, the returned ScheduleRuleset
            will be 0 = unoccupied, 1 = occupied. Otherwise the ScheduleRuleset
            will be the weighted fractional occupancy schedule.
        model_occ_hr_start: [Double] Starting decimal hour of whole building occupancy.
        model_occ_hr_end: [Double] Ending decimal hour of whole building occupancy.
        proportional_gain: [Double] Proportional gain constant (recommended 0.3 or less).
        switch_over_time: [Double] Time limitation for when the system can switch
            between heating and cooling.
    """
    zone_name = ems_friendly_name(zone.nameString())
    zone_timestep = model.getTimestep().numberOfTimestepsPerHour()

    if model.version() < openstudio.VersionString('3.1.1'):
        coil_cooling_radiant = \
            radiant_loop.coolingCoil().to_CoilCoolingLowTempRadiantVarFlow().get()
        coil_heating_radiant = \
            radiant_loop.heatingCoil().to_CoilHeatingLowTempRadiantVarFlow().get()
    else:
        coil_cooling_radiant = \
            radiant_loop.coolingCoil().get().to_CoilCoolingLowTempRadiantVarFlow().get()
        coil_heating_radiant = \
            radiant_loop.heatingCoil().get().to_CoilHeatingLowTempRadiantVarFlow().get()

    #####
    # Define radiant system parameters
    ####
    # set radiant system temperature and setpoint control type
    control_types = ('surfacefacetemperature', 'surfaceinteriortemperature')
    if not radiant_temperature_control_type.lower() in control_types:
        msg = 'Control sequences not compatible with "{}" radiant system control. ' \
            'Defaulting to "SurfaceFaceTemperature".'.format(radiant_temperature_control_type)
        print(msg)
        radiant_temperature_control_type = 'SurfaceFaceTemperature'

    radiant_loop.setTemperatureControlType(radiant_temperature_control_type)

    #####
    # List of schedule objects used to hold calculation results
    ####

    # get existing switchover time schedule or create one if needed
    sch_radiant_switchover = model.getScheduleRulesetByName('Radiant System Switchover')
    if sch_radiant_switchover.is_initialized():
        sch_radiant_switchover = sch_radiant_switchover.get()
    else:
        sch_radiant_switchover = create_constant_schedule_ruleset(
            model, switch_over_time, name='Radiant System Switchover',
            schedule_type_limit='Dimensionless')

    # set radiant system switchover schedule
    radiant_loop.setChangeoverDelayTimePeriodSchedule(
        sch_radiant_switchover.to_Schedule().get())

    # Calculated active slab heating and cooling temperature setpoint.
    # radiant system cooling control actuator
    sch_radiant_clgsetp = create_constant_schedule_ruleset(
        model, 26.0, name='{}_Sch_Radiant_ClgSetP'.format(zone_name),
        schedule_type_limit='Temperature')
    coil_cooling_radiant.setCoolingControlTemperatureSchedule(sch_radiant_clgsetp)
    cmd_cold_water_ctrl = openstudio_model.EnergyManagementSystemActuator(
        sch_radiant_clgsetp, 'Schedule:Year', 'Schedule Value')
    cmd_cold_water_ctrl.setName('{}_cmd_cold_water_ctrl'.format(zone_name))

    # radiant system heating control actuator
    sch_radiant_htgsetp = create_constant_schedule_ruleset(
        model, 20.0, name='{}_Sch_Radiant_HtgSetP'.format(zone_name),
        schedule_type_limit='Temperature')
    coil_heating_radiant.setHeatingControlTemperatureSchedule(sch_radiant_htgsetp)
    cmd_hot_water_ctrl = openstudio_model.EnergyManagementSystemActuator(
      sch_radiant_htgsetp, 'Schedule:Year', 'Schedule Value')
    cmd_hot_water_ctrl.setName('{}_cmd_hot_water_ctrl'.format(zone_name))

    # Calculated cooling setpoint error. Calculated from upper comfort limit minus
    # setpoint offset and 'measured' controlled zone temperature.
    sch_csp_error = create_constant_schedule_ruleset(
        model, 0.0, name='{}_Sch_CSP_Error'.format(zone_name),
        schedule_type_limit='Temperature')
    cmd_csp_error = openstudio_model.EnergyManagementSystemActuator(
        sch_csp_error, 'Schedule:Year', 'Schedule Value')
    cmd_csp_error.setName('{}_cmd_csp_error'.format(zone_name))

    # Calculated heating setpoint error. Calculated from lower comfort limit plus
    # setpoint offset and 'measured' controlled zone temperature.
    sch_hsp_error = create_constant_schedule_ruleset(
        model, 0.0, name='{}_Sch_HSP_Error'.format(zone_name),
        schedule_type_limit='Temperature')
    cmd_hsp_error = openstudio_model.EnergyManagementSystemActuator(
        sch_hsp_error, 'Schedule:Year', 'Schedule Value')
    cmd_hsp_error.setName('{}_cmd_hsp_error'.format(zone_name))

    #####
    # List of global variables used in EMS scripts
    ####

    # Proportional  gain constant (recommended 0.3 or less).
    prp_k = model.getEnergyManagementSystemGlobalVariableByName('prp_k')
    if prp_k.is_initialized():
        prp_k = prp_k.get()
    else:
        prp_k = openstudio_model.EnergyManagementSystemGlobalVariable(model, 'prp_k')

    # Upper slab temperature setpoint limit (recommended no higher than 29C (84F))
    upper_slab_sp_lim = model.getEnergyManagementSystemGlobalVariableByName(
       'upper_slab_sp_lim')
    if upper_slab_sp_lim.is_initialized():
        upper_slab_sp_lim = upper_slab_sp_lim.get()
    else:
        upper_slab_sp_lim = openstudio_model.EnergyManagementSystemGlobalVariable(
            model, 'upper_slab_sp_lim')

    # Lower slab temperature setpoint limit (recommended no lower than 19C (66F))
    lower_slab_sp_lim = model.getEnergyManagementSystemGlobalVariableByName(
       'lower_slab_sp_lim')
    if lower_slab_sp_lim.is_initialized():
        lower_slab_sp_lim = lower_slab_sp_lim.get()
    else:
        lower_slab_sp_lim = openstudio_model.EnergyManagementSystemGlobalVariable(
            model, 'lower_slab_sp_lim')

    # Temperature offset used as a safety factor for thermal control (recommend 0.5C (1F)).
    ctrl_temp_offset = model.getEnergyManagementSystemGlobalVariableByName(
        'ctrl_temp_offset')
    if ctrl_temp_offset.is_initialized():
        ctrl_temp_offset = ctrl_temp_offset.get()
    else:
        ctrl_temp_offset = openstudio_model.EnergyManagementSystemGlobalVariable(
            model, 'ctrl_temp_offset')

    # Hour where slab setpoint is to be changed
    hour_of_slab_sp_change = model.getEnergyManagementSystemGlobalVariableByName(
        'hour_of_slab_sp_change')
    if hour_of_slab_sp_change.is_initialized():
        hour_of_slab_sp_change = hour_of_slab_sp_change.get
    else:
        hour_of_slab_sp_change = openstudio_model.EnergyManagementSystemGlobalVariable(
            model, 'hour_of_slab_sp_change')

    #####
    # List of zone specific variables used in EMS scripts
    ####

    # Maximum 'measured' temperature in zone during occupied times.
    # Default setup uses mean air temperature.
    # Other possible choices are operative and mean radiant temperature.
    zone_max_ctrl_temp = openstudio_model.EnergyManagementSystemGlobalVariable(
        model, '{}_max_ctrl_temp'.format(zone_name))

    # Minimum 'measured' temperature in zone during occupied times. Default setup uses mean air temperature.
    # Other possible choices are operative and mean radiant temperature.
    zone_min_ctrl_temp = openstudio_model.EnergyManagementSystemGlobalVariable(
        model, '{}_min_ctrl_temp'.format(zone_name))

    #####
    # List of 'sensors' used in the EMS programs
    ####

    # Controlled zone temperature for the zone.
    zone_ctrl_temperature = openstudio_model.EnergyManagementSystemSensor(
        model, 'Zone Air Temperature')
    zone_ctrl_temperature.setName('{}_ctrl_temperature'.format(zone_name))
    zone_ctrl_temperature.setKeyName(zone.nameString())

    # check for zone thermostat and replace heat/cool schedules for radiant system control
    # if there is no zone thermostat, then create one
    zone_thermostat = zone.thermostatSetpointDualSetpoint()
    if zone_thermostat.is_initialized():
        zone_thermostat = zone_thermostat.get()
    else:
        zone_thermostat = openstudio_model.ThermostatSetpointDualSetpoint(model)
        zone_thermostat.setName('{}_Thermostat_DualSetpoint'.format(zone_name))

    # create new heating and cooling schedules to be used with all radiant systems
    zone_htg_thermostat = model.getScheduleRulesetByName('Radiant System Heating Setpoint')
    if zone_htg_thermostat.is_initialized():
        zone_htg_thermostat = zone_htg_thermostat.get()
    else:
        zone_htg_thermostat = create_constant_schedule_ruleset(
            model, 20.0, name='Radiant System Heating Setpoint',
            schedule_type_limit='Temperature')

    zone_clg_thermostat = model.getScheduleRulesetByName('Radiant System Cooling Setpoint')
    if zone_clg_thermostat.is_initialized():
        zone_clg_thermostat = zone_clg_thermostat.get()
    else:
        zone_clg_thermostat = create_constant_schedule_ruleset(
            model, 26.0, name='Radiant System Cooling Setpoint',
            schedule_type_limit='Temperature')

    # implement new heating and cooling schedules
    zone_thermostat.setHeatingSetpointTemperatureSchedule(zone_htg_thermostat)
    zone_thermostat.setCoolingSetpointTemperatureSchedule(zone_clg_thermostat)

    # Upper comfort limit for the zone. Taken from existing thermostat schedules in the zone.
    zone_upper_comfort_limit = openstudio_model.EnergyManagementSystemSensor(
        model, 'Schedule Value')
    zone_upper_comfort_limit.setName('{}_upper_comfort_limit'.format(zone_name))
    zone_upper_comfort_limit.setKeyName(zone_clg_thermostat.nameString())

    # Lower comfort limit for the zone. Taken from existing thermostat schedules in the zone.
    zone_lower_comfort_limit = openstudio_model.EnergyManagementSystemSensor(
        model, 'Schedule Value')
    zone_lower_comfort_limit.setName('{}_lower_comfort_limit'.format(zone_name))
    zone_lower_comfort_limit.setKeyName(zone_htg_thermostat.nameString())

    # Radiant system water flow rate used to determine if there is active
    # hydronic cooling in the radiant system.
    zone_rad_cool_operation = openstudio_model.EnergyManagementSystemSensor(
        model, 'System Node Mass Flow Rate')
    zone_rad_cool_operation.setName('{}_rad_cool_operation'.format(zone_name))
    cool_in = coil_cooling_radiant.to_StraightComponent().get().inletModelObject().get()
    zone_rad_cool_operation.setKeyName(cool_in.nameString())

    # Radiant system water flow rate used to determine if there is active
    # hydronic heating in the radiant system.
    zone_rad_heat_operation = openstudio_model.EnergyManagementSystemSensor(
        model, 'System Node Mass Flow Rate')
    zone_rad_heat_operation.setName('{}_rad_heat_operation'.format(zone_name))
    heat_in = coil_heating_radiant.to_StraightComponent().get().inletModelObject().get()
    zone_rad_heat_operation.setKeyName(heat_in.nameString())

    # Radiant system switchover delay time period schedule
    # used to determine if there is active hydronic cooling/heating in the radiant system.
    zone_rad_switch_over = model.getEnergyManagementSystemSensorByName('radiant_switch_over_time')

    if not zone_rad_switch_over.is_initialized():
        zone_rad_switch_over = openstudio_model.EnergyManagementSystemSensor(
            model, 'Schedule Value')
        zone_rad_switch_over.setName('radiant_switch_over_time')
        zone_rad_switch_over.setKeyName(sch_radiant_switchover.nameString())

    # Last 24 hours trend for radiant system in cooling mode.
    zone_rad_cool_operation_trend = openstudio_model.EnergyManagementSystemTrendVariable(
        model, zone_rad_cool_operation)
    zone_rad_cool_operation_trend.setName('{}_rad_cool_operation_trend'.format(zone_name))
    zone_rad_cool_operation_trend.setNumberOfTimestepsToBeLogged(zone_timestep * 48)

    # Last 24 hours trend for radiant system in heating mode.
    zone_rad_heat_operation_trend = openstudio_model.EnergyManagementSystemTrendVariable(
        model, zone_rad_heat_operation)
    zone_rad_heat_operation_trend.setName('{}_rad_heat_operation_trend'.format(zone_name))
    zone_rad_heat_operation_trend.setNumberOfTimestepsToBeLogged(zone_timestep * 48)

    # use zone occupancy objects for radiant system control if selected
    if use_zone_occupancy_for_control:
        # get annual occupancy schedule for zone
        occ_schedule = thermal_zone_get_occupancy_schedule(model, zone)
    else:
        occ_schedule = model.getScheduleRulesetByName(
            'Whole Building Radiant System Occupied Schedule')
        if occ_schedule.is_initialized():
            occ_schedule = occ_schedule.get()
        else:
            # create occupancy schedules
            occ_schedule = openstudio_model.ScheduleRuleset(model)
            occ_schedule.setName(
                'Whole Building Radiant System Occupied Schedule')

            start_hour = int(model_occ_hr_end)
            start_minute = int((model_occ_hr_end % 1) * 60)
            end_hour = int(model_occ_hr_start)
            end_minute = int((model_occ_hr_start % 1) * 60)

            if end_hour > start_hour:
                occ_schedule.defaultDaySchedule.addValue(
                    openstudio.Time(0, start_hour, start_minute, 0), 1.0)
                occ_schedule.defaultDaySchedule.addValue(
                    openstudio.Time(0, end_hour, end_minute, 0), 0.0)
                if end_hour < 24:
                    occ_schedule.defaultDaySchedule.addValue(
                        openstudio.Time(0, 24, 0, 0), 1.0)
            elif start_hour > end_hour:
                occ_schedule.defaultDaySchedule.addValue(
                    openstudio.Time(0, end_hour, end_minute, 0), 0.0)
                occ_schedule.defaultDaySchedule.addValue(
                    openstudio.Time(0, start_hour, start_minute, 0), 1.0)
                if start_hour < 24:
                    occ_schedule.defaultDaySchedule.addValue(
                        openstudio.Time(0, 24, 0, 0), 0.0)
            else:
                occ_schedule.defaultDaySchedule.addValue(
                    openstudio.Time(0, 24, 0, 0), 1.0)

    # create ems sensor for zone occupied status
    zone_occupied_status = \
        openstudio_model.EnergyManagementSystemSensor(model, 'Schedule Value')
    zone_occupied_status.setName('{}_occupied_status'.format(zone_name))
    zone_occupied_status.setKeyName(occ_schedule.nameString())

    # Last 24 hours trend for zone occupied status
    zone_occupied_status_trend = \
        openstudio_model.EnergyManagementSystemTrendVariable(model, zone_occupied_status)
    zone_occupied_status_trend.setName('{}_occupied_status_trend'.format(zone_name))
    zone_occupied_status_trend.setNumberOfTimestepsToBeLogged(zone_timestep * 48)

    #####
    # List of EMS programs to implement the proportional control for the radiant system.
    ####

    # Initialize global constant values used in EMS programs.
    set_constant_values_prg_body = \
        'SET prp_k              = {},\n' \
        'SET ctrl_temp_offset   = 0.5,\n' \
        'SET upper_slab_sp_lim  = 29,\n' \
        'SET lower_slab_sp_lim  = 19,\n' \
        'SET hour_of_slab_sp_change = 18'.format(proportional_gain)

    set_constant_values_prg = \
        model.getEnergyManagementSystemProgramByName('Set_Constant_Values')
    if set_constant_values_prg.is_initialized():
        set_constant_values_prg = set_constant_values_prg.get()
    else:
        set_constant_values_prg = openstudio_model.EnergyManagementSystemProgram(model)
        set_constant_values_prg.setName('Set_Constant_Values')
        set_constant_values_prg.setBody(set_constant_values_prg_body)

    # Initialize zone specific constant values used in EMS programs.
    set_constant_zone_values_prg_body = \
        'SET {zone_name}_max_ctrl_temp      = {zone_name}_lower_comfort_limit,\n' \
        'SET {zone_name}_min_ctrl_temp      = {zone_name}_upper_comfort_limit,\n' \
        'SET {zone_name}_cmd_csp_error      = 0,\n' \
        'SET {zone_name}_cmd_hsp_error      = 0,\n' \
        'SET {zone_name}_cmd_cold_water_ctrl = {zone_name}_upper_comfort_limit,\n' \
        'SET {zone_name}_cmd_hot_water_ctrl  = {zone_name}_lower_comfort_limit'.format(
            zone_name=zone_name)

    set_constant_zone_values_prg = openstudio_model.EnergyManagementSystemProgram(model)
    set_constant_zone_values_prg.setName('{}_Set_Constant_Values'.format(zone_name))
    set_constant_zone_values_prg.setBody(set_constant_zone_values_prg_body)

    # Calculate maximum and minimum 'measured' controlled temperature in the zone
    calculate_minmax_ctrl_temp_prg = openstudio_model.EnergyManagementSystemProgram(model)
    calculate_minmax_ctrl_temp_prg.setName('{}_Calculate_Extremes_In_Zone'.format(zone_name))
    calculate_minmax_ctrl_temp_prg_body = \
        'IF ({zone_name}_occupied_status >= {occ_threshold}),\n' \
        '    IF {zone_name}_ctrl_temperature > {zone_name}_max_ctrl_temp,\n' \
        '        SET {zone_name}_max_ctrl_temp = {zone_name}_ctrl_temperature,\n' \
        '    ENDIF,\n' \
        '    IF {zone_name}_ctrl_temperature < {zone_name}_min_ctrl_temp,\n' \
        '        SET {zone_name}_min_ctrl_temp = {zone_name}_ctrl_temperature,\n' \
        '    ENDIF,\n' \
        'ELSE,\n' \
        '    SET {zone_name}_max_ctrl_temp = {zone_name}_lower_comfort_limit,\n' \
        '    SET {zone_name}_min_ctrl_temp = {zone_name}_upper_comfort_limit,\n' \
        'ENDIF'.format(
            zone_name=zone_name, occ_threshold=occupied_percentage_threshold)

    calculate_minmax_ctrl_temp_prg.setBody(calculate_minmax_ctrl_temp_prg_body)

    # Calculate errors from comfort zone limits and 'measured' temperature in the zone.
    calculate_errors_from_comfort_prg = openstudio_model.EnergyManagementSystemProgram(model)
    calculate_errors_from_comfort_prg.setName(
        '{}_Calculate_Errors_From_Comfort'.format(zone_name))
    calculate_errors_from_comfort_prg_body = \
        'IF (CurrentTime == (hour_of_slab_sp_change - ZoneTimeStep)),\n' \
        '    SET {zone_name}_cmd_csp_error = ' \
        '({zone_name}_upper_comfort_limit - ctrl_temp_offset) - {zone_name}_max_ctrl_temp,\n' \
        '    SET {zone_name}_cmd_hsp_error = ' \
        '({zone_name}_lower_comfort_limit + ctrl_temp_offset) - {zone_name}_min_ctrl_temp,\n' \
        'ENDIF'.format(zone_name=zone_name)
    calculate_errors_from_comfort_prg.setBody(calculate_errors_from_comfort_prg_body)

    # Calculate the new active slab temperature setpoint for heating and cooling
    calculate_slab_ctrl_setpoint_prg = \
        openstudio_model.EnergyManagementSystemProgram(model)
    calculate_slab_ctrl_setpoint_prg.setName(
        '{}_Calculate_Slab_Ctrl_Setpoint'.format(zone_name))
    calculate_slab_ctrl_setpoint_prg_body = \
        'SET {z}_cont_cool_oper = ' \
        '@TrendSum {z}_rad_cool_operation_trend radiant_switch_over_time/ZoneTimeStep,\n' \
        'SET {z}_cont_heat_oper = ' \
        '@TrendSum {z}_rad_heat_operation_trend radiant_switch_over_time/ZoneTimeStep,\n' \
        'SET {z}_occupied_hours = @TrendSum {z}_occupied_status_trend 24/ZoneTimeStep,\n' \
        'IF ({z}_cont_cool_oper > 0) && ({z}_occupied_hours > 0) && ' \
        '(CurrentTime == hour_of_slab_sp_change),\n' \
        '    SET {z}_cmd_hot_water_ctrl = ' \
        '{z}_cmd_hot_water_ctrl + ({z}_cmd_csp_error*prp_k),\n' \
        'ELSEIF ({z}_cont_heat_oper > 0) && ({z}_occupied_hours > 0) && ' \
        '(CurrentTime == hour_of_slab_sp_change),\n' \
        '    SET {z}_cmd_hot_water_ctrl = ' \
        '{z}_cmd_hot_water_ctrl + ({z}_cmd_hsp_error*prp_k),\n' \
        'ELSE,\n' \
        '    SET {z}_cmd_hot_water_ctrl = {z}_cmd_hot_water_ctrl,\n' \
        'ENDIF,\n' \
        'IF ({z}_cmd_hot_water_ctrl < lower_slab_sp_lim),\n' \
        '    SET {z}_cmd_hot_water_ctrl = lower_slab_sp_lim,\n' \
        'ELSEIF ({z}_cmd_hot_water_ctrl > upper_slab_sp_lim),\n' \
        '    SET {z}_cmd_hot_water_ctrl = upper_slab_sp_lim,\n' \
        'ENDIF,\n' \
        'SET {z}_cmd_cold_water_ctrl = {z}_cmd_hot_water_ctrl + 0.01'.format(z=zone_name)

    calculate_slab_ctrl_setpoint_prg.setBody(calculate_slab_ctrl_setpoint_prg_body)

    #####
    # List of EMS program manager objects
    ####

    initialize_constant_parameters = \
        model.getEnergyManagementSystemProgramCallingManagerByName(
            'Initialize_Constant_Parameters')
    if initialize_constant_parameters.is_initialized():
        initialize_constant_parameters = initialize_constant_parameters.get()
        # add program if it does not exist in manager
        existing_program_names = []
        for prg in initialize_constant_parameters.programs():
            existing_program_names.append(prg.nameString().lower())
        if set_constant_values_prg.nameString().lower() not in existing_program_names:
            initialize_constant_parameters.addProgram(set_constant_values_prg)
    else:
        initialize_constant_parameters = \
            openstudio_model.EnergyManagementSystemProgramCallingManager(model)
        initialize_constant_parameters.setName('Initialize_Constant_Parameters')
        initialize_constant_parameters.setCallingPoint('BeginNewEnvironment')
        initialize_constant_parameters.addProgram(set_constant_values_prg)

    initialize_constant_parameters_after_warmup = \
        model.getEnergyManagementSystemProgramCallingManagerByName(
            'Initialize_Constant_Parameters_After_Warmup')
    if initialize_constant_parameters_after_warmup.is_initialized():
        initialize_constant_parameters_after_warmup = \
            initialize_constant_parameters_after_warmup.get()
        # add program if it does not exist in manager
        existing_program_names = []
        for prg in initialize_constant_parameters_after_warmup.programs():
            existing_program_names.append(prg.nameString().lower())
        if set_constant_values_prg.nameString().lower() not in existing_program_names:
            initialize_constant_parameters_after_warmup.addProgram(set_constant_values_prg)
    else:
        initialize_constant_parameters_after_warmup = \
            openstudio_model.EnergyManagementSystemProgramCallingManager(model)
        initialize_constant_parameters_after_warmup.setName(
            'Initialize_Constant_Parameters_After_Warmup')
        initialize_constant_parameters_after_warmup.setCallingPoint(
            'AfterNewEnvironmentWarmUpIsComplete')
        initialize_constant_parameters_after_warmup.addProgram(set_constant_values_prg)

    zone_initialize_constant_parameters = \
        openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    zone_initialize_constant_parameters.setName(
        '{}_Initialize_Constant_Parameters'.format(zone_name))
    zone_initialize_constant_parameters.setCallingPoint('BeginNewEnvironment')
    zone_initialize_constant_parameters.addProgram(set_constant_zone_values_prg)

    zone_initialize_constant_parameters_after_warmup = \
        openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    zone_initialize_constant_parameters_after_warmup.setName(
        '{}_Initialize_Constant_Parameters_After_Warmup'.format(zone_name))
    zone_initialize_constant_parameters_after_warmup.setCallingPoint(
        'AfterNewEnvironmentWarmUpIsComplete')
    zone_initialize_constant_parameters_after_warmup.addProgram(
        set_constant_zone_values_prg)

    average_building_temperature = \
        openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    average_building_temperature.setName(
        '{}_Average_Building_Temperature'.format(zone_name))
    average_building_temperature.setCallingPoint('EndOfZoneTimestepAfterZoneReporting')
    average_building_temperature.addProgram(calculate_minmax_ctrl_temp_prg)
    average_building_temperature.addProgram(calculate_errors_from_comfort_prg)

    programs_at_beginning_of_timestep = \
        openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    programs_at_beginning_of_timestep.setName(
        '{}_Programs_At_Beginning_Of_Timestep'.format(zone_name))
    programs_at_beginning_of_timestep.setCallingPoint('BeginTimestepBeforePredictor')
    programs_at_beginning_of_timestep.addProgram(calculate_slab_ctrl_setpoint_prg)

    #####
    # List of variables for output.
    ####

    zone_max_ctrl_temp_output = openstudio_model.EnergyManagementSystemOutputVariable(
        model, zone_max_ctrl_temp)
    zone_max_ctrl_temp_output.setName(
        '{} Maximum occupied temperature in zone'.format(zone_name))
    zone_min_ctrl_temp_output = openstudio_model.EnergyManagementSystemOutputVariable(
        model, zone_min_ctrl_temp)
    zone_min_ctrl_temp_output.setName(
        '{} Minimum occupied temperature in zone'.format(zone_name))


def model_add_radiant_basic_controls(
        model, zone, radiant_loop,
        radiant_temperature_control_type='SurfaceFaceTemperature',
        slab_setpoint_oa_control=False, switch_over_time=24.0,
        slab_sp_at_oat_low=73, slab_oat_low=65,
        slab_sp_at_oat_high=68, slab_oat_high=80):
    """Native EnergyPlus objects implement control for a single zone with a radiant system.

    Args:
        model: OpenStudio Model.
        zone: [OpenStudio::Model::ThermalZone>] zone to add radiant controls.
        radiant_loop: [OpenStudio::Model::ZoneHVACLowTempRadiantVarFlow>] radiant
            loop in thermal zone.
        radiant_temperature_control_type: [String] determines the controlled
            temperature for the radiant system. Options include the following.

            * SurfaceFaceTemperature
            * SurfaceInteriorTemperature

        slab_setpoint_oa_control: [Bool] True if slab setpoint is to be varied
            based on outdoor air temperature.
        switch_over_time: [Double] Time limitation for when the system can switch
            between heating and cooling.
        slab_sp_at_oat_low: [Double] radiant slab temperature setpoint, in F, at
            the outdoor high temperature.
        slab_oat_low: [Double] outdoor drybulb air temperature, in F, for low
            radiant slab setpoint.
        slab_sp_at_oat_high: [Double] radiant slab temperature setpoint, in F,
            at the outdoor low temperature.
        slab_oat_high: [Double] outdoor drybulb air temperature, in F, for high
            radiant slab setpoint.
    """
    zone_name = zone.nameString()
    if model.version() < openstudio.VersionString('3.1.1'):
        coil_cooling_radiant = \
            radiant_loop.coolingCoil().to_CoilCoolingLowTempRadiantVarFlow().get()
        coil_heating_radiant = \
            radiant_loop.heatingCoil().to_CoilHeatingLowTempRadiantVarFlow().get()
    else:
        coil_cooling_radiant = \
            radiant_loop.coolingCoil.get().to_CoilCoolingLowTempRadiantVarFlow().get()
        coil_heating_radiant = \
            radiant_loop.heatingCoil.get().to_CoilHeatingLowTempRadiantVarFlow().get()

    #####
    # Define radiant system parameters
    ####
    # set radiant system temperature and setpoint control type
    control_types = ('surfacefacetemperature', 'surfaceinteriortemperature')
    if not radiant_temperature_control_type.lower() in control_types:
        msg = 'Control sequences not compatible with "{}" radiant system control. ' \
            'Defaulting to "SurfaceFaceTemperature".'.format(radiant_temperature_control_type)
        print(msg)
        radiant_temperature_control_type = 'SurfaceFaceTemperature'

    radiant_loop.setTemperatureControlType(radiant_temperature_control_type)

    # get existing switchover time schedule or create one if needed
    sch_radiant_switchover = model.getScheduleRulesetByName('Radiant System Switchover')
    if sch_radiant_switchover.is_initialized():
        sch_radiant_switchover = sch_radiant_switchover.get()
    else:
        sch_radiant_switchover = create_constant_schedule_ruleset(
            model, switch_over_time, name='Radiant System Switchover',
            schedule_type_limit='Dimensionless')

    # set radiant system switchover schedule
    radiant_loop.setChangeoverDelayTimePeriodSchedule(
       sch_radiant_switchover.to_Schedule().get())

    if slab_setpoint_oa_control:
        schedule_interval = model.getScheduleByName(
            'Sch_Radiant_SlabSetP_Based_On_Rolling_Mean_OAT')
        if schedule_interval.is_initialized() and \
                schedule_interval.get().to_ScheduleFixedInterval().is_initialized():
            schedule_interval = schedule_interval.get().to_ScheduleFixedInterval().get()
            coil_heating_radiant.setHeatingControlTemperatureSchedule(schedule_interval)
            coil_cooling_radiant.setCoolingControlTemperatureSchedule(schedule_interval)
        else:
            # get weather file from model
            weather_file = model.getWeatherFile()
            if weather_file.initialized():
                # get annual outdoor dry bulb temperature
                epw_data = weather_file.file().get().data()
                annual_oat = [dat.dryBulbTemperature().get() for dat in epw_data]

                # calculate a nhrs rolling average from annual outdoor dry bulb temperature
                nhrs = 24
                oat_rolling_average = []
                annual_oat = list(annual_oat[-nhrs:]) + annual_oat
                for i in range(nhrs, len(annual_oat)):
                    avg = sum(annual_oat[i - nhrs: i]) / nhrs
                    oat_rolling_average.append(round(avg, 2))

                # use rolling average to calculate slab setpoint temperature

                # convert temperature from IP to SI units
                slab_sp_at_oat_low_si = \
                    TEMPERATURE.to_unit([slab_sp_at_oat_low], 'C', 'F')[0]
                slab_oat_low_si = TEMPERATURE.to_unit([slab_oat_low], 'C', 'F')[0]
                slab_sp_at_oat_high_si = \
                    TEMPERATURE.to_unit([slab_sp_at_oat_high], 'C', 'F')[0]
                slab_oat_high_si = TEMPERATURE.to_unit([slab_oat_high], 'C', 'F')[0]

                # calculate relationship between slab setpoint and slope
                slope_num = slab_sp_at_oat_high_si - slab_sp_at_oat_low_si
                slope_den = slab_oat_high_si - slab_oat_low_si
                sp_and_oat_slope = round(slope_num / slope_den, 4)

                slab_setpoint = []
                for e in oat_rolling_average:
                    sl_pt = slab_sp_at_oat_low_si + ((e - slab_oat_low_si) * sp_and_oat_slope)
                    slab_setpoint.append(round(sl_pt, 1))

                # input upper limits on slab setpoint
                slab_sp_upper_limit = max([slab_sp_at_oat_high_si, slab_sp_at_oat_low_si])
                slab_sp_upper_limit = round(slab_sp_upper_limit, 1)
                slab_setpoint = [e if e < slab_sp_upper_limit else slab_sp_upper_limit
                                 for e in slab_setpoint]

                # input lower limits on slab setpoint
                slab_sp_lower_limit = min([slab_sp_at_oat_high_si, slab_sp_at_oat_low_si])
                slab_sp_lower_limit = round(slab_sp_lower_limit, 1)
                slab_setpoint = [e if e > slab_sp_lower_limit else slab_sp_lower_limit
                                 for e in slab_setpoint]

                # convert to timeseries
                interval = openstudio.Time(0, 0, 60)
                start_date = model.makeDate(1, 1)
                series_values = openstudio.Vector(len(slab_setpoint))
                for i, val in enumerate(slab_setpoint):
                    series_values[i] = val
                time_series = openstudio.TimeSeries(start_date, interval, series_values, 'C')

                # create fixed interval schedule for slab setpoint
                schedule_interval = openstudio_model.ScheduleFixedInterval(model)
                schedule_interval.setName('Sch_Radiant_SlabSetP_Based_On_Rolling_Mean_OAT')
                schedule_interval.setTimeSeries(time_series)
                sch_type_limits_obj = create_schedule_type_limits(
                    model, standard_schedule_type_limit='Temperature')
                schedule_interval.setScheduleTypeLimits(sch_type_limits_obj)

                # assign slab setpoint schedule
                coil_heating_radiant.setHeatingControlTemperatureSchedule(schedule_interval)
                coil_cooling_radiant.setCoolingControlTemperatureSchedule(schedule_interval)
            else:
                msg = 'Model does not have a weather file associated with it. ' \
                    'Define to implement slab setpoint based on outdoor weather.'
                raise ValueError(msg)
    else:
        slab_setpoint = 22  # constant setpoint for the slab surface temperature
        # radiant system cooling control setpoint
        sch_radiant_clgsetp = create_constant_schedule_ruleset(
            model, slab_setpoint + 0.1, name='{}_Sch_Radiant_ClgSetP'.format(zone_name),
            schedule_type_limit='Temperature')
        coil_cooling_radiant.setCoolingControlTemperatureSchedule(sch_radiant_clgsetp)

        # radiant system heating control setpoint
        sch_radiant_htgsetp = create_constant_schedule_ruleset(
            model, slab_setpoint, name='{}_Sch_Radiant_HtgSetP'.format(zone_name),
            schedule_type_limit='Temperature')
        coil_heating_radiant.setHeatingControlTemperatureSchedule(sch_radiant_htgsetp)
