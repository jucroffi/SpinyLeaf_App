# coding=utf-8
"""Methods to write Honeybee-energy SimulationParameters to OpenStudio."""
from __future__ import division
import os

from ladybug.epw import EPW
from ladybug.stat import STAT
from ladybug.designday import ASHRAEClearSky, ASHRAETau

from honeybee_openstudio.openstudio import OSModel, OSOutputVariable, \
    OSDesignDay, OSRunPeriodControlSpecialDays, OSMonthOfYear, \
    openstudio, os_path, os_vector_len, openstudio_model

STANDARD_MAP = {
    'DOE_Ref_Pre_1980': 'DOE Ref Pre-1980',
    'DOE_Ref_1980_2004': 'DOE Ref 1980-2004',
    'ASHRAE_2004': '90.1-2004',
    'ASHRAE_2007': '90.1-2007',
    'ASHRAE_2010': '90.1-2010',
    'ASHRAE_2013': '90.1-2013',
    'ASHRAE_2016': '90.1-2016',
    'ASHRAE_2019': '90.1-2019'
}


def assign_epw_to_model(epw_file, os_model, set_climate_zone=False):
    """Assign an EPW file to an OpenStudio Model.

    Args:
        epw_file: Path to an EPW file to be assigned to the OpenStudio Model.
        os_model: The OpenStudio Model object to which the EPW will be assigned.
        set_climate_zone: A boolean to note whether the EPW file should be used
            to set the OpenStudio Model's ASHRAE climate zone if it has not
            already been specified.
    """
    # set the EPW file if specified and possibly use it to assign the climate zone
    assert os.path.isfile(epw_file), 'No EPW file was found at: {}'.format(epw_file)
    assert epw_file.lower().endswith('.epw'), \
        'The file does not have .epw extension: {}'.format(epw_file)
    full_path = os.path.abspath(epw_file)
    try:
        os_epw = openstudio.EpwFile(os_path(full_path))
    except AttributeError:  # older bindings with no EPW module
        return os_model
    openstudio_model.WeatherFile.setWeatherFile(os_model, os_epw)
    # set the ASHRAE climate zone if requested
    if set_climate_zone:
        climate_zone_objs = os_model.getClimateZones()
        ashrae_zones = climate_zone_objs.getClimateZones('ASHRAE')
        if os_vector_len(ashrae_zones) == 0:
            # first see if there is a STAT file from which a better zone can be pulled
            cz_set = False
            stat_path = epw_file[-4:] + '.stat'
            if os.path.isfile(stat_path):
                stat = STAT(stat_path)
                if stat.ashrae_climate_zone is not None:
                    climate_zone_objs.setClimateZone('ASHRAE', stat.ashrae_climate_zone)
                    cz_set = True
            if not cz_set:  # use temperature in the EPW to estimate the climate zone
                epw = EPW(full_path)
                climate_zone_objs.setClimateZone('ASHRAE', epw.ashrae_climate_zone)
    return os_model


def design_day_to_openstudio(design_day, os_model):
    """Convert Ladybug DesignDay to OpenStudio DesignDay.

    Args:
        design_day: A Ladybug DesignDay to be translated to OpenStudio.
        os_model: The OpenStudio Model object.
    """
    # create the DesignDay object
    os_des_day = OSDesignDay(os_model)
    os_des_day.setName(design_day.name)
    os_des_day.setDayType(design_day.day_type)
    # set the DryBulbCondition properties
    db_cond = design_day.dry_bulb_condition
    os_des_day.setMaximumDryBulbTemperature(db_cond.dry_bulb_max)
    os_des_day.setDailyDryBulbTemperatureRange(db_cond.dry_bulb_range)
    # set the HumidityCondition properties
    humid_cond = design_day.humidity_condition
    os_des_day.setHumidityConditionType(humid_cond.humidity_type)
    if humid_cond.humidity_type == 'HumidityRatio':
        os_des_day.setHumidityRatioAtMaximumDryBulb(humid_cond.humidity_value)
    elif humid_cond.humidity_type == 'Enthalpy':
        os_des_day.setEnthalpyAtMaximumDryBulb(humid_cond.humidity_value)
    else:
        os_des_day.setWetBulbOrDewPointAtMaximumDryBulb(humid_cond.humidity_value)
    os_des_day.setBarometricPressure(humid_cond.barometric_pressure)
    os_des_day.setRainIndicator(humid_cond.rain)
    os_des_day.setSnowIndicator(humid_cond.snow_on_ground)
    # set the WindCondition properties
    wind_cond = design_day.wind_condition
    os_des_day.setWindSpeed(wind_cond.wind_speed)
    os_des_day.setWindDirection(wind_cond.wind_direction)
    # set the SkyCondition properties
    sky_cond = design_day.sky_condition
    os_des_day.setMonth(sky_cond.date.month)
    os_des_day.setDayOfMonth(sky_cond.date.day)
    os_des_day.setDaylightSavingTimeIndicator(sky_cond.daylight_savings)
    if isinstance(sky_cond, ASHRAEClearSky):
        os_des_day.setSolarModelIndicator('ASHRAEClearSky')
        os_des_day.setSkyClearness(sky_cond.clearness)
    elif isinstance(sky_cond, ASHRAETau):
        model_type = 'ASHRAETau2017' if sky_cond.use_2017 else 'ASHRAETau'
        os_des_day.setSolarModelIndicator(model_type)
        os_des_day.setAshraeClearSkyOpticalDepthForBeamIrradiance(sky_cond.tau_b)
        os_des_day.setAshraeClearSkyOpticalDepthForDiffuseIrradiance(sky_cond.tau_d)
    return os_des_day


def sizing_to_openstudio(sizing, os_model):
    """Convert Honeybee SizingParameter to OpenStudio SizingParameters.

    Args:
        sizing: A Honeybee-energy SizingParameter to be translated to OpenStudio.
        os_model: The OpenStudio Model object.
    """
    os_sizing_par = os_model.getSizingParameters()
    os_sizing_par.setHeatingSizingFactor(sizing.heating_factor)
    os_sizing_par.setCoolingSizingFactor(sizing.cooling_factor)
    for des_day in sizing.design_days:
        design_day_to_openstudio(des_day, os_model)
    building = os_model.getBuilding()
    if sizing.efficiency_standard is not None:
        std_gem_standard = STANDARD_MAP[sizing.efficiency_standard]
        building.setStandardsTemplate(std_gem_standard)
    if sizing.building_type is not None:
        building.setStandardsBuildingType(sizing.building_type)
    else:
        building.setStandardsBuildingType('MediumOffice')
    if sizing.climate_zone is not None:
        climate_zone_objs = os_model.getClimateZones()
        climate_zone_objs.setClimateZone('ASHRAE', sizing.climate_zone)
    return os_sizing_par


def simulation_control_to_openstudio(control, os_model):
    """Convert Honeybee SimulationControl to OpenStudio SimulationControl.

    Args:
        control: A Honeybee-energy SimulationControl to be translated to OpenStudio.
        os_model: The OpenStudio Model object.
    """
    os_sim_control = os_model.getSimulationControl()
    os_sim_control.setDoZoneSizingCalculation(control.do_zone_sizing)
    os_sim_control.setDoSystemSizingCalculation(control.do_system_sizing)
    os_sim_control.setDoPlantSizingCalculation(control.do_plant_sizing)
    os_sim_control.setRunSimulationforWeatherFileRunPeriods(control.run_for_run_periods)
    os_sim_control.setRunSimulationforSizingPeriods(control.run_for_sizing_periods)
    return os_sim_control


def shadow_calculation_to_openstudio(shadow_calc, os_model):
    """Convert Honeybee ShadowCalculation to OpenStudio ShadowCalculation.

    Args:
        shadow_calc: A Honeybee-energy ShadowCalculation to be translated to OpenStudio.
        os_model: The OpenStudio Model object.
    """
    os_sim_control = os_model.getSimulationControl()
    os_sim_control.setSolarDistribution(shadow_calc.solar_distribution)
    os_shadow_calc = os_model.getShadowCalculation()
    os_shadow_calc.setShadingCalculationMethod(shadow_calc.calculation_method)
    os_shadow_calc.setShadingCalculationUpdateFrequencyMethod(
        shadow_calc.calculation_update_method)
    os_shadow_calc.setShadingCalculationUpdateFrequency(
        shadow_calc.calculation_frequency)
    os_shadow_calc.setMaximumFiguresInShadowOverlapCalculations(
        shadow_calc.maximum_figures)
    return os_shadow_calc


def simulation_output_to_openstudio(sim_output, os_model):
    """Convert Honeybee SimulationOutput to a list of OpenStudio OutputVariables.

    Args:
        sim_output: A Honeybee-energy SimulationOutput to be translated to OpenStudio.
        os_model: The OpenStudio Model object.
    """
    # set outputs for the simulation
    os_outputs = []
    for output in sim_output.outputs:
        os_output = OSOutputVariable(output, os_model)
        os_output.setReportingFrequency(sim_output.reporting_frequency)
        os_outputs.append(os_output)
    # set the summary reports for the simulation
    if len(sim_output.summary_reports) != 0:
        os_report = os_model.getOutputTableSummaryReports()
        for report in sim_output.summary_reports:
            os_report.addSummaryReport(report)
    # set the not met setpoint reporting tolerances
    os_unmet_tol = os_model.getOutputControlReportingTolerances()
    set_tol = sim_output.unmet_setpoint_tolerance
    os_unmet_tol.setToleranceforTimeHeatingSetpointNotMet(set_tol)
    os_unmet_tol.setToleranceforTimeCoolingSetpointNotMet(set_tol)
    return os_outputs


def run_period_to_openstudio(run_period, os_model):
    """Convert Honeybee RunPeriod to OpenStudio RunPeriod.

    Args:
        run_period: A Honeybee-energy RunPeriod to be translated to OpenStudio.
        os_model: The OpenStudio Model object.
    """
    # set the characteristics of the year
    os_model.setDayofWeekforStartDay(run_period.start_day_of_week)
    if run_period.is_leap_year:
        os_model.setIsLeapYear(True)
    # set the run period start and end dates
    os_run_period = os_model.getRunPeriod()
    os_run_period.setBeginMonth(run_period.start_date.month)
    os_run_period.setBeginDayOfMonth(run_period.start_date.day)
    os_run_period.setEndMonth(run_period.end_date.month)
    os_run_period.setEndDayOfMonth(run_period.end_date.day)
    # set the holidays
    if run_period.holidays is not None:
        for hol in run_period.holidays:
            os_hol_month = OSMonthOfYear(hol.month)
            os_hol = OSRunPeriodControlSpecialDays(os_hol_month, hol.day, os_model)
            os_hol.setDuration(1)
            os_hol.setSpecialDayType('Holiday')
    # set the daylight savings time
    if run_period.daylight_saving_time is not None:
        dls = run_period.daylight_saving_time
        os_dl_saving = os_model.getRunPeriodControlDaylightSavingTime()
        os_dls_s_month = OSMonthOfYear(dls.start_date.month)
        os_dl_saving.setStartDate(os_dls_s_month, dls.start_date.day)
        os_dls_e_month = OSMonthOfYear(dls.end_date.month)
        os_dl_saving.setEndDate(os_dls_e_month, dls.end_date.day)
    return os_run_period


def simulation_parameter_to_openstudio(sim_par, seed_model=None):
    """Convert Honeybee SimulationParameter to an OpenStudio model.

    Args:
        sim_par: A Honeybee-energy SimulationParameter to be translated to OpenStudio.
        seed_model: An optional OpenStudio Model object to which the Honeybee
            Model will be added. If None, a new OpenStudio Model will be
            initialized within this method. (Default: None).
    """
    # create the OpenStudio model object
    os_model = OSModel() if seed_model is None else seed_model
    # translate all of the sub-classes to OpenStudio
    simulation_output_to_openstudio(sim_par.output, os_model)
    run_period_to_openstudio(sim_par.run_period, os_model)
    sizing_to_openstudio(sim_par.sizing_parameter, os_model)
    simulation_control_to_openstudio(sim_par.simulation_control, os_model)
    shadow_calculation_to_openstudio(sim_par.shadow_calculation, os_model)
    # translate the miscellaneous attributes to openstudio
    os_timestep = os_model.getTimestep()
    os_timestep.setNumberOfTimestepsPerHour(sim_par.timestep)
    os_building = os_model.getBuilding()
    os_building.setNorthAxis(sim_par.north_angle)
    os_site = os_model.getSite()
    os_site.setTerrain(sim_par.terrain_type)
    # set the water mains to use the EPW temperature
    os_water_mains = os_model.getSiteWaterMainsTemperature()
    os_water_mains.setCalculationMethod('CorrelationFromWeatherFile')
    return os_model
