# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/
Prototype.AirConditionerVariableRefrigerantFlow.rb
"""
from __future__ import division

from ladybug.datatype.temperature import Temperature

from honeybee_openstudio.openstudio import openstudio, openstudio_model
from .schedule import model_add_schedule

TEMPERATURE = Temperature()


def create_air_conditioner_variable_refrigerant_flow(
        model, name='VRF System', schedule=None, type=None,
        cooling_cop=4.287, heating_cop=4.147, heat_recovery=True,
        defrost_strategy='Resistive', condenser_type='AirCooled', condenser_loop=None,
        master_zone=None, priority_control_type='LoadPriority'):
    """Prototype AirConditionerVariableRefrigerantFlow object.

    Enters in default curves for coil by type of coil.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        name: [String] the name of the system, or None in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or None in which case
            default to always on.
        type: [String] the type of unit to reference for the correct curve set.
        cooling_cop: [Double] rated cooling coefficient of performance.
        heating_cop: [Double] rated heating coefficient of performance.
        heat_recovery: [Boolean] does the unit have heat recovery.
        defrost_strategy: [String] type of defrost strategy. Options are
            ReverseCycle or Resistive.
        condenser_type: [String] type of condenser.
            options are AirCooled (default), WaterCooled, and EvaporativelyCooled.
            if WaterCooled, the user most include a condenser_loop.
        master_zone: [<OpenStudio::Model::ThermalZone>] master control zone to
            switch between heating and cooling.
        priority_control_type: [String] Type of master thermostat priority
            control type. Options are LoadPriority, ZonePriority,
            ThermostatOffsetPriority, MasterThermostatPriority.
    """
    vrf_outdoor_unit = openstudio_model.AirConditionerVariableRefrigerantFlow(model)

    # set name
    name = 'VRF System' if name is None else name
    vrf_outdoor_unit.setName(name)

    # set availability schedule
    if schedule is None:  # default always on
        availability_schedule = model.alwaysOnDiscreteSchedule()
    elif isinstance(schedule, str):
        if schedule == 'alwaysOffDiscreteSchedule':
            availability_schedule = model.alwaysOffDiscreteSchedule()
        elif schedule == 'alwaysOnDiscreteSchedule':
            availability_schedule = model.alwaysOnDiscreteSchedule()
        else:
            availability_schedule = model_add_schedule(model, schedule)
    else:  # assume that it is an actual schedule object
        availability_schedule = schedule
    vrf_outdoor_unit.setAvailabilitySchedule(availability_schedule)

    # set cops
    if model.version() < openstudio.VersionString('2.9.0'):
        vrf_outdoor_unit.setRatedCoolingCOP(cooling_cop)
    else:
        vrf_outdoor_unit.setGrossRatedCoolingCOP(cooling_cop)
    vrf_outdoor_unit.setRatedHeatingCOP(heating_cop)

    # heat recovery
    vrf_outdoor_unit.setHeatPumpWasteHeatRecovery(heat_recovery)

    # defrost strategy
    vrf_outdoor_unit.setDefrostStrategy(defrost_strategy)

    # defaults
    vrf_outdoor_unit.setMinimumOutdoorTemperatureinCoolingMode(-15.0)
    vrf_outdoor_unit.setMaximumOutdoorTemperatureinCoolingMode(50.0)
    vrf_outdoor_unit.setMinimumOutdoorTemperatureinHeatingMode(-25.0)
    vrf_outdoor_unit.setMaximumOutdoorTemperatureinHeatingMode(16.1)
    vrf_outdoor_unit.setMinimumOutdoorTemperatureinHeatRecoveryMode(-10.0)
    vrf_outdoor_unit.setMaximumOutdoorTemperatureinHeatRecoveryMode(27.2)
    vrf_outdoor_unit.setEquivalentPipingLengthusedforPipingCorrectionFactorinCoolingMode(30.48)
    vrf_outdoor_unit.setEquivalentPipingLengthusedforPipingCorrectionFactorinHeatingMode(30.48)
    vrf_outdoor_unit.setVerticalHeightusedforPipingCorrectionFactor(10.668)

    # condenser type
    if condenser_type == 'WaterCooled':
        vrf_outdoor_unit.setString(56, condenser_type)
        # require condenser_loop
        if not condenser_loop:
            msg = 'Must specify condenser_loop for vrf_outdoor_unit if WaterCooled'
            raise ValueError(msg)
        condenser_loop.addDemandBranchForComponent(vrf_outdoor_unit)
    elif condenser_type == 'EvaporativelyCooled':
        vrf_outdoor_unit.setString(56, condenser_type)

    # set master zone
    if master_zone is not None and master_zone.to_ThermalZone().is_initialized():
        vrf_outdoor_unit.setZoneforMasterThermostatLocation(master_zone)
        vrf_outdoor_unit.setMasterThermostatPriorityControlType(priority_control_type)

    vrf_cool_cap_f_of_low_temp = None
    vrf_cool_cap_ratio_boundary = None
    vrf_cool_cap_f_of_high_temp = None
    vrf_cool_eir_f_of_low_temp = None
    vrf_cool_eir_ratio_boundary = None
    vrf_cool_eir_f_of_high_temp = None
    vrf_cooling_eir_low_plr = None
    vrf_cooling_eir_high_plr = None
    vrf_cooling_comb_ratio = None
    vrf_cooling_cplffplr = None
    vrf_heat_cap_f_of_low_temp = None
    vrf_heat_cap_ratio_boundary = None
    vrf_heat_cap_f_of_high_temp = None
    vrf_heat_eir_f_of_low_temp = None
    vrf_heat_eir_boundary = None
    vrf_heat_eir_f_of_high_temp = None
    vrf_heating_eir_low_plr = None
    vrf_heating_eir_hi_plr = None
    vrf_heating_comb_ratio = None
    vrf_heating_cplffplr = None
    vrf_defrost_eir_f_of_temp = None

    # curve sets
    if type == 'OS default':
        pass  # use OS default curves

    else:  # default curve set
        # based on DAIKINREYQ 120 on BCL

        # Cooling Capacity Ratio Modifier Function of Low Temperature Curve
        vrf_cool_cap_f_of_low_temp = openstudio_model.CurveBiquadratic(model)
        vrf_cool_cap_f_of_low_temp.setName('vrf_cool_cap_f_of_low_temp')
        vrf_cool_cap_f_of_low_temp.setCoefficient1Constant(-1.69653019339465)
        vrf_cool_cap_f_of_low_temp.setCoefficient2x(0.207248180531939)
        vrf_cool_cap_f_of_low_temp.setCoefficient3xPOW2(-0.00343146229659024)
        vrf_cool_cap_f_of_low_temp.setCoefficient4y(0.016381597419714)
        vrf_cool_cap_f_of_low_temp.setCoefficient5yPOW2(-6.7387172629965e-05)
        vrf_cool_cap_f_of_low_temp.setCoefficient6xTIMESY(-0.000849848402870241)
        vrf_cool_cap_f_of_low_temp.setMinimumValueofx(13.9)
        vrf_cool_cap_f_of_low_temp.setMaximumValueofx(23.9)
        vrf_cool_cap_f_of_low_temp.setMinimumValueofy(-5.0)
        vrf_cool_cap_f_of_low_temp.setMaximumValueofy(43.3)
        vrf_cool_cap_f_of_low_temp.setMinimumCurveOutput(0.59)
        vrf_cool_cap_f_of_low_temp.setMaximumCurveOutput(1.33)

        # Cooling Capacity Ratio Boundary Curve
        vrf_cool_cap_ratio_boundary = openstudio_model.CurveCubic(model)
        vrf_cool_cap_ratio_boundary.setName('vrf_cool_cap_ratio_boundary')
        vrf_cool_cap_ratio_boundary.setCoefficient1Constant(25.73)
        vrf_cool_cap_ratio_boundary.setCoefficient2x(-0.03150043)
        vrf_cool_cap_ratio_boundary.setCoefficient3xPOW2(-0.01416595)
        vrf_cool_cap_ratio_boundary.setCoefficient4xPOW3(0.0)
        vrf_cool_cap_ratio_boundary.setMinimumValueofx(11.0)
        vrf_cool_cap_ratio_boundary.setMaximumValueofx(30.0)

        # Cooling Capacity Ratio Modifier Function of High Temperature Curve
        vrf_cool_cap_f_of_high_temp = openstudio_model.CurveBiquadratic(model)
        vrf_cool_cap_f_of_high_temp.setName('vrf_cool_cap_f_of_high_temp')
        vrf_cool_cap_f_of_high_temp.setCoefficient1Constant(0.6867358)
        vrf_cool_cap_f_of_high_temp.setCoefficient2x(0.0207631)
        vrf_cool_cap_f_of_high_temp.setCoefficient3xPOW2(0.0005447)
        vrf_cool_cap_f_of_high_temp.setCoefficient4y(-0.0016218)
        vrf_cool_cap_f_of_high_temp.setCoefficient5yPOW2(-4.259e-07)
        vrf_cool_cap_f_of_high_temp.setCoefficient6xTIMESY(-0.0003392)
        vrf_cool_cap_f_of_high_temp.setMinimumValueofx(15.0)
        vrf_cool_cap_f_of_high_temp.setMaximumValueofx(24.0)
        vrf_cool_cap_f_of_high_temp.setMinimumValueofy(16.0)
        vrf_cool_cap_f_of_high_temp.setMaximumValueofy(43.0)

        # Cooling Energy Input Ratio Modifier Function of Low Temperature Curve
        vrf_cool_eir_f_of_low_temp = openstudio_model.CurveBiquadratic(model)
        vrf_cool_eir_f_of_low_temp.setName('vrf_cool_eir_f_of_low_temp')
        vrf_cool_eir_f_of_low_temp.setCoefficient1Constant(-1.61908214818635)
        vrf_cool_eir_f_of_low_temp.setCoefficient2x(0.185964818731756)
        vrf_cool_eir_f_of_low_temp.setCoefficient3xPOW2(-0.00389610393381592)
        vrf_cool_eir_f_of_low_temp.setCoefficient4y(-0.00901995326324613)
        vrf_cool_eir_f_of_low_temp.setCoefficient5yPOW2(0.00030340007815629)
        vrf_cool_eir_f_of_low_temp.setCoefficient6xTIMESY(0.000476048529099348)
        vrf_cool_eir_f_of_low_temp.setMinimumValueofx(13.9)
        vrf_cool_eir_f_of_low_temp.setMaximumValueofx(23.9)
        vrf_cool_eir_f_of_low_temp.setMinimumValueofy(-5.0)
        vrf_cool_eir_f_of_low_temp.setMaximumValueofy(43.3)
        vrf_cool_eir_f_of_low_temp.setMinimumCurveOutput(0.27)
        vrf_cool_eir_f_of_low_temp.setMaximumCurveOutput(1.15)

        # Cooling Energy Input Ratio Boundary Curve
        vrf_cool_eir_ratio_boundary = openstudio_model.CurveCubic(model)
        vrf_cool_eir_ratio_boundary.setName('vrf_cool_eir_ratio_boundary')
        vrf_cool_eir_ratio_boundary.setCoefficient1Constant(25.73473775)
        vrf_cool_eir_ratio_boundary.setCoefficient2x(-0.03150043)
        vrf_cool_eir_ratio_boundary.setCoefficient3xPOW2(-0.01416595)
        vrf_cool_eir_ratio_boundary.setCoefficient4xPOW3(0.0)
        vrf_cool_eir_ratio_boundary.setMinimumValueofx(15.0)
        vrf_cool_eir_ratio_boundary.setMaximumValueofx(24.0)

        # Cooling Energy Input Ratio Modifier Function of High Temperature Curve
        vrf_cool_eir_f_of_high_temp = openstudio_model.CurveBiquadratic(model)
        vrf_cool_eir_f_of_high_temp.setName('vrf_cool_eir_f_of_high_temp')
        vrf_cool_eir_f_of_high_temp.setCoefficient1Constant(-1.4395110176)
        vrf_cool_eir_f_of_high_temp.setCoefficient2x(0.1619850459)
        vrf_cool_eir_f_of_high_temp.setCoefficient3xPOW2(-0.0034911781)
        vrf_cool_eir_f_of_high_temp.setCoefficient4y(0.0269442645)
        vrf_cool_eir_f_of_high_temp.setCoefficient5yPOW2(0.0001346163)
        vrf_cool_eir_f_of_high_temp.setCoefficient6xTIMESY(-0.0006714941)
        vrf_cool_eir_f_of_high_temp.setMinimumValueofx(15.0)
        vrf_cool_eir_f_of_high_temp.setMaximumValueofx(23.9)
        vrf_cool_eir_f_of_high_temp.setMinimumValueofy(16.8)
        vrf_cool_eir_f_of_high_temp.setMaximumValueofy(43.3)

        # Cooling Energy Input Ratio Modifier Function of Low Part-Load Ratio Curve
        vrf_cooling_eir_low_plr = openstudio_model.CurveCubic(model)
        vrf_cooling_eir_low_plr.setName('vrf_cool_eir_f_of_low_temp')
        vrf_cooling_eir_low_plr.setCoefficient1Constant(0.0734992169827752)
        vrf_cooling_eir_low_plr.setCoefficient2x(0.334783365234032)
        vrf_cooling_eir_low_plr.setCoefficient3xPOW2(0.591613015486343)
        vrf_cooling_eir_low_plr.setCoefficient4xPOW3(0.0)
        vrf_cooling_eir_low_plr.setMinimumValueofx(0.25)
        vrf_cooling_eir_low_plr.setMaximumValueofx(1.0)
        vrf_cooling_eir_low_plr.setMinimumCurveOutput(0.0)
        vrf_cooling_eir_low_plr.setMaximumCurveOutput(1.0)

        # Cooling Energy Input Ratio Modifier Function of High Part-Load Ratio Curve
        vrf_cooling_eir_high_plr = openstudio_model.CurveCubic(model)
        vrf_cooling_eir_high_plr.setName('vrf_cooling_eir_high_plr')
        vrf_cooling_eir_high_plr.setCoefficient1Constant(1.0)
        vrf_cooling_eir_high_plr.setCoefficient2x(0.0)
        vrf_cooling_eir_high_plr.setCoefficient3xPOW2(0.0)
        vrf_cooling_eir_high_plr.setCoefficient4xPOW3(0.0)
        vrf_cooling_eir_high_plr.setMinimumValueofx(1.0)
        vrf_cooling_eir_high_plr.setMaximumValueofx(1.5)

        # Cooling Combination Ratio Correction Factor Curve
        vrf_cooling_comb_ratio = openstudio_model.CurveCubic(model)
        vrf_cooling_comb_ratio.setName('vrf_cooling_comb_ratio')
        vrf_cooling_comb_ratio.setCoefficient1Constant(0.24034)
        vrf_cooling_comb_ratio.setCoefficient2x(-0.21873)
        vrf_cooling_comb_ratio.setCoefficient3xPOW2(1.97941)
        vrf_cooling_comb_ratio.setCoefficient4xPOW3(-1.02636)
        vrf_cooling_comb_ratio.setMinimumValueofx(0.5)
        vrf_cooling_comb_ratio.setMaximumValueofx(2.0)
        vrf_cooling_comb_ratio.setMinimumCurveOutput(0.5)
        vrf_cooling_comb_ratio.setMaximumCurveOutput(1.056)

        # Cooling Part-Load Fraction Correlation Curve
        vrf_cooling_cplffplr = openstudio_model.CurveCubic(model)
        vrf_cooling_cplffplr.setName('vrf_cooling_cplffplr')
        vrf_cooling_cplffplr.setCoefficient1Constant(0.85)
        vrf_cooling_cplffplr.setCoefficient2x(0.15)
        vrf_cooling_cplffplr.setCoefficient3xPOW2(0.0)
        vrf_cooling_cplffplr.setCoefficient4xPOW3(0.0)
        vrf_cooling_cplffplr.setMinimumValueofx(1.0)
        vrf_cooling_cplffplr.setMaximumValueofx(1.0)

        # Heating Capacity Ratio Modifier Function of Low Temperature Curve Name
        vrf_heat_cap_f_of_low_temp = openstudio_model.CurveBiquadratic(model)
        vrf_heat_cap_f_of_low_temp.setName('vrf_heat_cap_f_of_low_temp')
        vrf_heat_cap_f_of_low_temp.setCoefficient1Constant(0.983220174655636)
        vrf_heat_cap_f_of_low_temp.setCoefficient2x(0.0157167577703294)
        vrf_heat_cap_f_of_low_temp.setCoefficient3xPOW2(-0.000835032422884084)
        vrf_heat_cap_f_of_low_temp.setCoefficient4y(0.0522939264581759)
        vrf_heat_cap_f_of_low_temp.setCoefficient5yPOW2(-0.000531556035364549)
        vrf_heat_cap_f_of_low_temp.setCoefficient6xTIMESY(-0.00190605953116024)
        vrf_heat_cap_f_of_low_temp.setMinimumValueofx(16.1)
        vrf_heat_cap_f_of_low_temp.setMaximumValueofx(23.9)
        vrf_heat_cap_f_of_low_temp.setMinimumValueofy(-25.0)
        vrf_heat_cap_f_of_low_temp.setMaximumValueofy(13.3)
        vrf_heat_cap_f_of_low_temp.setMinimumCurveOutput(0.515151515151515)
        vrf_heat_cap_f_of_low_temp.setMaximumCurveOutput(1.2)

        # Heating Capacity Ratio Boundary Curve Name
        vrf_heat_cap_ratio_boundary = openstudio_model.CurveCubic(model)
        vrf_heat_cap_ratio_boundary.setName('vrf_heat_cap_ratio_boundary')
        vrf_heat_cap_ratio_boundary.setCoefficient1Constant(58.577)
        vrf_heat_cap_ratio_boundary.setCoefficient2x(-3.0255)
        vrf_heat_cap_ratio_boundary.setCoefficient3xPOW2(0.0193)
        vrf_heat_cap_ratio_boundary.setCoefficient4xPOW3(0.0)
        vrf_heat_cap_ratio_boundary.setMinimumValueofx(15)
        vrf_heat_cap_ratio_boundary.setMaximumValueofx(23.9)

        # Heating Capacity Ratio Modifier Function of High Temperature Curve Name
        vrf_heat_cap_f_of_high_temp = openstudio_model.CurveBiquadratic(model)
        vrf_heat_cap_f_of_high_temp.setName('vrf_heat_cap_f_of_high_temp')
        vrf_heat_cap_f_of_high_temp.setCoefficient1Constant(2.5859872368)
        vrf_heat_cap_f_of_high_temp.setCoefficient2x(-0.0953227101)
        vrf_heat_cap_f_of_high_temp.setCoefficient3xPOW2(0.0009553288)
        vrf_heat_cap_f_of_high_temp.setCoefficient4y(0.0)
        vrf_heat_cap_f_of_high_temp.setCoefficient5yPOW2(0.0)
        vrf_heat_cap_f_of_high_temp.setCoefficient6xTIMESY(0.0)
        vrf_heat_cap_f_of_high_temp.setMinimumValueofx(21.1)
        vrf_heat_cap_f_of_high_temp.setMaximumValueofx(27.2)
        vrf_heat_cap_f_of_high_temp.setMinimumValueofy(-944)
        vrf_heat_cap_f_of_high_temp.setMaximumValueofy(15)

        # Heating Energy Input Ratio Modifier Function of Low Temperature Curve Name
        vrf_heat_eir_f_of_low_temp = openstudio_model.CurveBiquadratic(model)
        vrf_heat_eir_f_of_low_temp.setName('vrf_heat_eir_f_of_low_temp')
        vrf_heat_eir_f_of_low_temp.setCoefficient1Constant(0.756830029796909)
        vrf_heat_eir_f_of_low_temp.setCoefficient2x(0.0457499799042671)
        vrf_heat_eir_f_of_low_temp.setCoefficient3xPOW2(-0.00136357240431388)
        vrf_heat_eir_f_of_low_temp.setCoefficient4y(0.0554884599902023)
        vrf_heat_eir_f_of_low_temp.setCoefficient5yPOW2(-0.00120700875497686)
        vrf_heat_eir_f_of_low_temp.setCoefficient6xTIMESY(-0.00303329271420931)
        vrf_heat_eir_f_of_low_temp.setMinimumValueofx(16.1)
        vrf_heat_eir_f_of_low_temp.setMaximumValueofx(23.9)
        vrf_heat_eir_f_of_low_temp.setMinimumValueofy(-25.0)
        vrf_heat_eir_f_of_low_temp.setMaximumValueofy(13.3)
        vrf_heat_eir_f_of_low_temp.setMinimumCurveOutput(0.7)
        vrf_heat_eir_f_of_low_temp.setMaximumCurveOutput(1.184)

        # Heating Energy Input Ratio Boundary Curve Name
        vrf_heat_eir_boundary = openstudio_model.CurveCubic(model)
        vrf_heat_eir_boundary.setName('vrf_heat_eir_boundary')
        vrf_heat_eir_boundary.setCoefficient1Constant(58.577)
        vrf_heat_eir_boundary.setCoefficient2x(-3.0255)
        vrf_heat_eir_boundary.setCoefficient3xPOW2(0.0193)
        vrf_heat_eir_boundary.setCoefficient4xPOW3(0.0)
        vrf_heat_eir_boundary.setMinimumValueofx(15.0)
        vrf_heat_eir_boundary.setMaximumValueofx(23.9)

        # Heating Energy Input Ratio Modifier Function of High Temperature Curve Name
        vrf_heat_eir_f_of_high_temp = openstudio_model.CurveBiquadratic(model)
        vrf_heat_eir_f_of_high_temp.setName('vrf_heat_eir_f_of_high_temp')
        vrf_heat_eir_f_of_high_temp.setCoefficient1Constant(1.3885703646)
        vrf_heat_eir_f_of_high_temp.setCoefficient2x(-0.0229771462)
        vrf_heat_eir_f_of_high_temp.setCoefficient3xPOW2(0.000537274)
        vrf_heat_eir_f_of_high_temp.setCoefficient4y(-0.0273936962)
        vrf_heat_eir_f_of_high_temp.setCoefficient5yPOW2(0.0004030426)
        vrf_heat_eir_f_of_high_temp.setCoefficient6xTIMESY(-5.9786e-05)
        vrf_heat_eir_f_of_high_temp.setMinimumValueofx(21.1)
        vrf_heat_eir_f_of_high_temp.setMaximumValueofx(27.2)
        vrf_heat_eir_f_of_high_temp.setMinimumValueofy(0.0)
        vrf_heat_eir_f_of_high_temp.setMaximumValueofy(1.0)

        # Heating Performance Curve Outdoor Temperature Type
        vrf_outdoor_unit.setHeatingPerformanceCurveOutdoorTemperatureType(
            'WetBulbTemperature')

        # Heating Energy Input Ratio Modifier Function of Low Part-Load Ratio Curve Name
        vrf_heating_eir_low_plr = openstudio_model.CurveCubic(model)
        vrf_heating_eir_low_plr.setName('vrf_heating_eir_low_plr')
        vrf_heating_eir_low_plr.setCoefficient1Constant(0.0724906507105475)
        vrf_heating_eir_low_plr.setCoefficient2x(0.658189977561701)
        vrf_heating_eir_low_plr.setCoefficient3xPOW2(0.269259536275246)
        vrf_heating_eir_low_plr.setCoefficient4xPOW3(0.0)
        vrf_heating_eir_low_plr.setMinimumValueofx(0.25)
        vrf_heating_eir_low_plr.setMaximumValueofx(1.0)
        vrf_heating_eir_low_plr.setMinimumCurveOutput(0.0)
        vrf_heating_eir_low_plr.setMaximumCurveOutput(1.0)

        # Heating Energy Input Ratio Modifier Function of High Part-Load Ratio Curve Name
        vrf_heating_eir_hi_plr = openstudio_model.CurveCubic(model)
        vrf_heating_eir_hi_plr.setName('vrf_heating_eir_hi_plr')
        vrf_heating_eir_hi_plr.setCoefficient1Constant(1.0)
        vrf_heating_eir_hi_plr.setCoefficient2x(0.0)
        vrf_heating_eir_hi_plr.setCoefficient3xPOW2(0.0)
        vrf_heating_eir_hi_plr.setCoefficient4xPOW3(0.0)
        vrf_heating_eir_hi_plr.setMinimumValueofx(1.0)
        vrf_heating_eir_hi_plr.setMaximumValueofx(1.5)

        # Heating Combination Ratio Correction Factor Curve Name
        vrf_heating_comb_ratio = openstudio_model.CurveCubic(model)
        vrf_heating_comb_ratio.setName('vrf_heating_comb_ratio')
        vrf_heating_comb_ratio.setCoefficient1Constant(0.62115)
        vrf_heating_comb_ratio.setCoefficient2x(-1.55798)
        vrf_heating_comb_ratio.setCoefficient3xPOW2(3.36817)
        vrf_heating_comb_ratio.setCoefficient4xPOW3(-1.4224)
        vrf_heating_comb_ratio.setMinimumValueofx(0.5)
        vrf_heating_comb_ratio.setMaximumValueofx(2.0)
        vrf_heating_comb_ratio.setMinimumCurveOutput(0.5)
        vrf_heating_comb_ratio.setMaximumCurveOutput(1.155)

        # Heating Part-Load Fraction Correlation Curve Name
        vrf_heating_cplffplr = openstudio_model.CurveCubic(model)
        vrf_heating_cplffplr.setName('vrf_heating_cplffplr')
        vrf_heating_cplffplr.setCoefficient1Constant(0.85)
        vrf_heating_cplffplr.setCoefficient2x(0.15)
        vrf_heating_cplffplr.setCoefficient3xPOW2(0.0)
        vrf_heating_cplffplr.setCoefficient4xPOW3(0.0)
        vrf_heating_cplffplr.setMinimumValueofx(1.0)
        vrf_heating_cplffplr.setMaximumValueofx(1.0)

        # Defrost Energy Input Ratio Modifier Function of Temperature Curve
        vrf_defrost_eir_f_of_temp = openstudio_model.CurveBiquadratic(model)
        vrf_defrost_eir_f_of_temp.setName('vrf_defrost_eir_f_of_temp')
        vrf_defrost_eir_f_of_temp.setCoefficient1Constant(-1.61908214818635)
        vrf_defrost_eir_f_of_temp.setCoefficient2x(0.185964818731756)
        vrf_defrost_eir_f_of_temp.setCoefficient3xPOW2(-0.00389610393381592)
        vrf_defrost_eir_f_of_temp.setCoefficient4y(-0.00901995326324613)
        vrf_defrost_eir_f_of_temp.setCoefficient5yPOW2(0.00030340007815629)
        vrf_defrost_eir_f_of_temp.setCoefficient6xTIMESY(0.000476048529099348)
        vrf_defrost_eir_f_of_temp.setMinimumValueofx(13.9)
        vrf_defrost_eir_f_of_temp.setMaximumValueofx(23.9)
        vrf_defrost_eir_f_of_temp.setMinimumValueofy(-5.0)
        vrf_defrost_eir_f_of_temp.setMaximumValueofy(50.0)
        vrf_defrost_eir_f_of_temp.setMinimumCurveOutput(0.27)
        vrf_defrost_eir_f_of_temp.setMaximumCurveOutput(1.155)

        # set defrost control
        vrf_outdoor_unit.setDefrostStrategy('ReverseCycle')
        vrf_outdoor_unit.setDefrostControl('OnDemand')

    if vrf_cool_cap_f_of_low_temp is not None:
        vrf_outdoor_unit.setCoolingCapacityRatioModifierFunctionofLowTemperatureCurve(
            vrf_cool_cap_f_of_low_temp)
    if vrf_cool_cap_ratio_boundary is not None:
        vrf_outdoor_unit.setCoolingCapacityRatioBoundaryCurve(
            vrf_cool_cap_ratio_boundary)
    if vrf_cool_cap_f_of_high_temp is not None:
        vrf_outdoor_unit.setCoolingCapacityRatioModifierFunctionofHighTemperatureCurve(
            vrf_cool_cap_f_of_high_temp)
    if vrf_cool_eir_f_of_low_temp is not None:
        vrf_outdoor_unit.setCoolingEnergyInputRatioModifierFunctionofLowTemperatureCurve(
            vrf_cool_eir_f_of_low_temp)
    if vrf_cool_eir_ratio_boundary is not None:
        vrf_outdoor_unit.setCoolingEnergyInputRatioBoundaryCurve(
            vrf_cool_eir_ratio_boundary)
    if vrf_cool_eir_f_of_high_temp is not None:
        vrf_outdoor_unit.setCoolingEnergyInputRatioModifierFunctionofHighTemperatureCurve(
            vrf_cool_eir_f_of_high_temp)
    if vrf_cooling_eir_low_plr is not None:
        vrf_outdoor_unit.setCoolingEnergyInputRatioModifierFunctionofLowPartLoadRatioCurve(
            vrf_cooling_eir_low_plr)
    if vrf_cooling_eir_high_plr is not None:
        vrf_outdoor_unit.setCoolingEnergyInputRatioModifierFunctionofHighPartLoadRatioCurve(
            vrf_cooling_eir_high_plr)
    if vrf_cooling_comb_ratio is not None:
        vrf_outdoor_unit.setCoolingCombinationRatioCorrectionFactorCurve(
            vrf_cooling_comb_ratio)
    if vrf_cooling_cplffplr is not None:
        vrf_outdoor_unit.setCoolingPartLoadFractionCorrelationCurve(
            vrf_cooling_cplffplr)
    if vrf_heat_cap_f_of_low_temp is not None:
        vrf_outdoor_unit.setHeatingCapacityRatioModifierFunctionofLowTemperatureCurve(
            vrf_heat_cap_f_of_low_temp)
    if vrf_heat_cap_ratio_boundary is not None:
        vrf_outdoor_unit.setHeatingCapacityRatioBoundaryCurve(
            vrf_heat_cap_ratio_boundary)
    if vrf_heat_cap_f_of_high_temp is not None:
        vrf_outdoor_unit.setHeatingCapacityRatioModifierFunctionofHighTemperatureCurve(
            vrf_heat_cap_f_of_high_temp)
    if vrf_heat_eir_f_of_low_temp is not None:
        vrf_outdoor_unit.setHeatingEnergyInputRatioModifierFunctionofLowTemperatureCurve(
            vrf_heat_eir_f_of_low_temp)
    if vrf_heat_eir_boundary is not None:
        vrf_outdoor_unit.setHeatingEnergyInputRatioBoundaryCurve(vrf_heat_eir_boundary)
    if vrf_heat_eir_f_of_high_temp is not None:
        vrf_outdoor_unit.setHeatingEnergyInputRatioModifierFunctionofHighTemperatureCurve(
            vrf_heat_eir_f_of_high_temp)
    if vrf_heating_eir_low_plr is not None:
        vrf_outdoor_unit.setHeatingEnergyInputRatioModifierFunctionofLowPartLoadRatioCurve(
            vrf_heating_eir_low_plr)
    if vrf_heating_eir_hi_plr is not None:
        vrf_outdoor_unit.setHeatingEnergyInputRatioModifierFunctionofHighPartLoadRatioCurve(
            vrf_heating_eir_hi_plr)
    if vrf_heating_comb_ratio is not None:
        vrf_outdoor_unit.setHeatingCombinationRatioCorrectionFactorCurve(
            vrf_heating_comb_ratio)
    if vrf_heating_cplffplr is not None:
        vrf_outdoor_unit.setHeatingPartLoadFractionCorrelationCurve(vrf_heating_cplffplr)
    if vrf_defrost_eir_f_of_temp is not None:
        vrf_outdoor_unit.setDefrostEnergyInputRatioModifierFunctionofTemperatureCurve(
            vrf_defrost_eir_f_of_temp)

    return vrf_outdoor_unit
