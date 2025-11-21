# coding=utf-8
"""Modules taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.CoilCoolingWater.rb
"""
from __future__ import division

from honeybee_openstudio.openstudio import openstudio, openstudio_model

from .utilities import create_curve_biquadratic, create_curve_quadratic, \
    convert_curve_biquadratic
from .schedule import model_add_schedule


def create_coil_cooling_water(
        model, chilled_water_loop, air_loop_node=None, name='Clg Coil', schedule=None,
        design_inlet_water_temperature=None, design_inlet_air_temperature=None,
        design_outlet_air_temperature=None):
    clg_coil = openstudio_model.CoilCoolingWater(model)

    # add to chilled water loop
    chilled_water_loop.addDemandBranchForComponent(clg_coil)

    # add to air loop if specified
    if air_loop_node is not None:
        clg_coil.addToNode(air_loop_node)

    # set coil name
    name = 'Clg Coil' if name is None else name
    clg_coil.setName(name)

    # set coil availability schedule
    if schedule is None:  # default always on
        coil_availability_schedule = model.alwaysOnDiscreteSchedule()
    elif isinstance(schedule, str):
        if schedule == 'alwaysOffDiscreteSchedule':
            coil_availability_schedule = model.alwaysOffDiscreteSchedule()
        elif schedule == 'alwaysOnDiscreteSchedule':
            coil_availability_schedule = model.alwaysOnDiscreteSchedule()
        else:
            coil_availability_schedule = model_add_schedule(model, schedule)
    else:  # assume that it is an actual schedule object
        coil_availability_schedule = schedule
    clg_coil.setAvailabilitySchedule(coil_availability_schedule)

    # rated temperatures
    if design_inlet_water_temperature is None:
        clg_coil.autosizeDesignInletWaterTemperature()
    else:
        clg_coil.setDesignInletWaterTemperature(design_inlet_water_temperature)
    if design_inlet_air_temperature is not None:
        clg_coil.setDesignInletAirTemperature(design_inlet_air_temperature)
    if design_outlet_air_temperature is not None:
        clg_coil.setDesignOutletAirTemperature(design_outlet_air_temperature)

    # defaults
    clg_coil.setHeatExchangerConfiguration('CrossFlow')

    # coil controller properties
    # @note These inputs will get overwritten if addToNode or addDemandBranchForComponent
    # is called on the htg_coil object after this
    clg_coil_controller = clg_coil.controllerWaterCoil().get()
    clg_coil_controller.setName('{} Controller'.format(clg_coil.nameString()))
    clg_coil_controller.setAction('Reverse')
    clg_coil_controller.setMinimumActuatedFlow(0.0)

    return clg_coil


def create_coil_cooling_dx_single_speed(
        model, air_loop_node=None, name='1spd DX Clg Coil', schedule=None,
        type=None, cop=None):
    """Prototype CoilCoolingDXSingleSpeed object.

    Enters in default curves for coil by type of coil

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or nil in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or nil in which case
            default to always on.
        type: [String] the type of single speed DX coil to reference the
            correct curve set.
        cop: [Double] rated cooling coefficient of performance.
    """
    clg_coil = openstudio_model.CoilCoolingDXSingleSpeed(model)

    # add to air loop if specified
    if air_loop_node is not None:
        clg_coil.addToNode(air_loop_node)

    # set coil name
    clg_coil.setName(name)

    # set coil availability schedule
    if schedule is None:  # default always on
        coil_availability_schedule = model.alwaysOnDiscreteSchedule()
    elif isinstance(schedule, str):
        if schedule == 'alwaysOffDiscreteSchedule':
            coil_availability_schedule = model.alwaysOffDiscreteSchedule()
        elif schedule == 'alwaysOnDiscreteSchedule':
            coil_availability_schedule = model.alwaysOnDiscreteSchedule()
        else:
            coil_availability_schedule = model_add_schedule(model, schedule)
    else:  # assume that it is an actual schedule object
        coil_availability_schedule = schedule
    clg_coil.setAvailabilitySchedule(coil_availability_schedule)

    # set coil cop
    if cop is not None:
        clg_coil.setRatedCOP(cop)

    clg_cap_f_of_temp = None
    clg_cap_f_of_flow = None
    clg_energy_input_ratio_f_of_temp = None
    clg_energy_input_ratio_f_of_flow = None
    clg_part_load_ratio = None

    # curve sets
    if type == 'OS default':
        pass  # use OS defaults

    elif type == 'Heat Pump':
        # "PSZ-AC_Unitary_PackagecoolCapFT"
        clg_cap_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_cap_f_of_temp.setCoefficient1Constant(0.766956)
        clg_cap_f_of_temp.setCoefficient2x(0.0107756)
        clg_cap_f_of_temp.setCoefficient3xPOW2(-0.0000414703)
        clg_cap_f_of_temp.setCoefficient4y(0.00134961)
        clg_cap_f_of_temp.setCoefficient5yPOW2(-0.000261144)
        clg_cap_f_of_temp.setCoefficient6xTIMESY(0.000457488)
        clg_cap_f_of_temp.setMinimumValueofx(12.78)
        clg_cap_f_of_temp.setMaximumValueofx(23.89)
        clg_cap_f_of_temp.setMinimumValueofy(21.1)
        clg_cap_f_of_temp.setMaximumValueofy(46.1)

        clg_cap_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_cap_f_of_flow.setCoefficient1Constant(0.8)
        clg_cap_f_of_flow.setCoefficient2x(0.2)
        clg_cap_f_of_flow.setCoefficient3xPOW2(0.0)
        clg_cap_f_of_flow.setMinimumValueofx(0.5)
        clg_cap_f_of_flow.setMaximumValueofx(1.5)

        clg_energy_input_ratio_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_energy_input_ratio_f_of_temp.setCoefficient1Constant(0.297145)
        clg_energy_input_ratio_f_of_temp.setCoefficient2x(0.0430933)
        clg_energy_input_ratio_f_of_temp.setCoefficient3xPOW2(-0.000748766)
        clg_energy_input_ratio_f_of_temp.setCoefficient4y(0.00597727)
        clg_energy_input_ratio_f_of_temp.setCoefficient5yPOW2(0.000482112)
        clg_energy_input_ratio_f_of_temp.setCoefficient6xTIMESY(-0.000956448)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofx(12.78)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofx(23.89)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofy(21.1)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofy(46.1)

        clg_energy_input_ratio_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_energy_input_ratio_f_of_flow.setCoefficient1Constant(1.156)
        clg_energy_input_ratio_f_of_flow.setCoefficient2x(-0.1816)
        clg_energy_input_ratio_f_of_flow.setCoefficient3xPOW2(0.0256)
        clg_energy_input_ratio_f_of_flow.setMinimumValueofx(0.5)
        clg_energy_input_ratio_f_of_flow.setMaximumValueofx(1.5)

        clg_part_load_ratio = openstudio_model.CurveQuadratic(model)
        clg_part_load_ratio.setCoefficient1Constant(0.85)
        clg_part_load_ratio.setCoefficient2x(0.15)
        clg_part_load_ratio.setCoefficient3xPOW2(0.0)
        clg_part_load_ratio.setMinimumValueofx(0.0)
        clg_part_load_ratio.setMaximumValueofx(1.0)

    elif type == 'PSZ-AC':
        # Defaults to "DOE Ref DX Clg Coil Cool-Cap-fT"
        clg_cap_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_cap_f_of_temp.setCoefficient1Constant(0.9712123)
        clg_cap_f_of_temp.setCoefficient2x(-0.015275502)
        clg_cap_f_of_temp.setCoefficient3xPOW2(0.0014434524)
        clg_cap_f_of_temp.setCoefficient4y(-0.00039321)
        clg_cap_f_of_temp.setCoefficient5yPOW2(-0.0000068364)
        clg_cap_f_of_temp.setCoefficient6xTIMESY(-0.0002905956)
        clg_cap_f_of_temp.setMinimumValueofx(-100.0)
        clg_cap_f_of_temp.setMaximumValueofx(100.0)
        clg_cap_f_of_temp.setMinimumValueofy(-100.0)
        clg_cap_f_of_temp.setMaximumValueofy(100.0)

        clg_cap_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_cap_f_of_flow.setCoefficient1Constant(1.0)
        clg_cap_f_of_flow.setCoefficient2x(0.0)
        clg_cap_f_of_flow.setCoefficient3xPOW2(0.0)
        clg_cap_f_of_flow.setMinimumValueofx(-100.0)
        clg_cap_f_of_flow.setMaximumValueofx(100.0)

        # "DOE Ref DX Clg Coil Cool-EIR-fT",
        clg_energy_input_ratio_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_energy_input_ratio_f_of_temp.setCoefficient1Constant(0.28687133)
        clg_energy_input_ratio_f_of_temp.setCoefficient2x(0.023902164)
        clg_energy_input_ratio_f_of_temp.setCoefficient3xPOW2(-0.000810648)
        clg_energy_input_ratio_f_of_temp.setCoefficient4y(0.013458546)
        clg_energy_input_ratio_f_of_temp.setCoefficient5yPOW2(0.0003389364)
        clg_energy_input_ratio_f_of_temp.setCoefficient6xTIMESY(-0.0004870044)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofx(-100.0)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofx(100.0)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofy(-100.0)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofy(100.0)

        clg_energy_input_ratio_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_energy_input_ratio_f_of_flow.setCoefficient1Constant(1.0)
        clg_energy_input_ratio_f_of_flow.setCoefficient2x(0.0)
        clg_energy_input_ratio_f_of_flow.setCoefficient3xPOW2(0.0)
        clg_energy_input_ratio_f_of_flow.setMinimumValueofx(-100.0)
        clg_energy_input_ratio_f_of_flow.setMaximumValueofx(100.0)

        # "DOE Ref DX Clg Coil Cool-PLF-fPLR"
        clg_part_load_ratio = openstudio_model.CurveQuadratic(model)
        clg_part_load_ratio.setCoefficient1Constant(0.90949556)
        clg_part_load_ratio.setCoefficient2x(0.09864773)
        clg_part_load_ratio.setCoefficient3xPOW2(-0.00819488)
        clg_part_load_ratio.setMinimumValueofx(0.0)
        clg_part_load_ratio.setMaximumValueofx(1.0)
        clg_part_load_ratio.setMinimumCurveOutput(0.7)
        clg_part_load_ratio.setMaximumCurveOutput(1.0)

    elif type == 'Window AC':
        # Performance curves
        # From Frigidaire 10.7 EER unit in Winkler et. al. Lab Testing of Window ACs (2013)
        # @note These coefficients are in SI UNITS
        cool_cap_ft_coeffs_si = [0.6405, 0.01568, 0.0004531,
                                 0.001615, -0.0001825, 0.00006614]
        cool_eir_ft_coeffs_si = [2.287, -0.1732, 0.004745, 0.01662, 0.000484, -0.001306]
        cool_cap_fflow_coeffs = [0.887, 0.1128, 0]
        cool_eir_fflow_coeffs = [1.763, -0.6081, 0]
        cool_plf_fplr_coeffs = [0.78, 0.22, 0]

        # Make the curves
        clg_cap_f_of_temp = create_curve_biquadratic(
            model, cool_cap_ft_coeffs_si, 'RoomAC-Cap-fT', 0, 100, 0, 100, None, None)
        clg_cap_f_of_flow = create_curve_quadratic(
            model, cool_cap_fflow_coeffs, 'RoomAC-Cap-fFF', 0, 2, 0, 2,
            is_dimensionless=True)
        clg_energy_input_ratio_f_of_temp = create_curve_biquadratic(
            model, cool_eir_ft_coeffs_si, 'RoomAC-EIR-fT', 0, 100, 0, 100, None, None)
        clg_energy_input_ratio_f_of_flow = create_curve_quadratic(
            model, cool_eir_fflow_coeffs, 'RoomAC-EIR-fFF', 0, 2, 0, 2,
            is_dimensionless=True)
        clg_part_load_ratio = create_curve_quadratic(
            model, cool_plf_fplr_coeffs, 'RoomAC-PLF-fPLR', 0, 1, 0, 1,
            is_dimensionless=True)

    elif type == 'Residential Central AC':
        # Performance curves
        # These coefficients are in IP UNITS
        cool_cap_ft_coeffs_ip = [3.670270705, -0.098652414, 0.000955906,
                                 0.006552414, -0.0000156, -0.000131877]
        cool_eir_ft_coeffs_ip = [-3.302695861, 0.137871531, -0.001056996,
                                 -0.012573945, 0.000214638, -0.000145054]
        cool_cap_fflow_coeffs = [0.718605468, 0.410099989, -0.128705457]
        cool_eir_fflow_coeffs = [1.32299905, -0.477711207, 0.154712157]
        cool_plf_fplr_coeffs = [0.8, 0.2, 0]

        # Convert coefficients from IP to SI
        cool_cap_ft_coeffs_si = convert_curve_biquadratic(cool_cap_ft_coeffs_ip)
        cool_eir_ft_coeffs_si = convert_curve_biquadratic(cool_eir_ft_coeffs_ip)

        # Make the curves
        clg_cap_f_of_temp = create_curve_biquadratic(
            model, cool_cap_ft_coeffs_si, 'AC-Cap-fT', 0, 100, 0, 100, None, None)
        clg_cap_f_of_flow = create_curve_quadratic(
            model, cool_cap_fflow_coeffs, 'AC-Cap-fFF', 0, 2, 0, 2, is_dimensionless=True)
        clg_energy_input_ratio_f_of_temp = create_curve_biquadratic(
            model, cool_eir_ft_coeffs_si, 'AC-EIR-fT', 0, 100, 0, 100, None, None)
        clg_energy_input_ratio_f_of_flow = create_curve_quadratic(
            model, cool_eir_fflow_coeffs, 'AC-EIR-fFF', 0, 2, 0, 2, is_dimensionless=True)
        clg_part_load_ratio = create_curve_quadratic(
            model, cool_plf_fplr_coeffs, 'AC-PLF-fPLR', 0, 1, 0, 1, is_dimensionless=True)

    elif type == 'Residential Central ASHP':
        # Performance curves
        # These coefficients are in IP UNITS
        cool_cap_ft_coeffs_ip = [3.68637657, -0.098352478, 0.000956357,
                                 0.005838141, -0.0000127, -0.000131702]
        cool_eir_ft_coeffs_ip = [-3.437356399, 0.136656369, -0.001049231,
                                 -0.0079378, 0.000185435, -0.0001441]
        cool_cap_fflow_coeffs = [0.718664047, 0.41797409, -0.136638137]
        cool_eir_fflow_coeffs = [1.143487507, -0.13943972, -0.004047787]
        cool_plf_fplr_coeffs = [0.8, 0.2, 0]

        # Convert coefficients from IP to SI
        cool_cap_ft_coeffs_si = convert_curve_biquadratic(cool_cap_ft_coeffs_ip)
        cool_eir_ft_coeffs_si = convert_curve_biquadratic(cool_eir_ft_coeffs_ip)

        # Make the curves
        clg_cap_f_of_temp = create_curve_biquadratic(
            model, cool_cap_ft_coeffs_si, 'Cool-Cap-fT', 0, 100, 0, 100, None, None)
        clg_cap_f_of_flow = create_curve_quadratic(
            model, cool_cap_fflow_coeffs, 'Cool-Cap-fFF', 0, 2, 0, 2,
            is_dimensionless=True)
        clg_energy_input_ratio_f_of_temp = create_curve_biquadratic(
            model, cool_eir_ft_coeffs_si, 'Cool-EIR-fT', 0, 100, 0, 100, None, None)
        clg_energy_input_ratio_f_of_flow = create_curve_quadratic(
            model, cool_eir_fflow_coeffs, 'Cool-EIR-fFF', 0, 2, 0, 2,
            is_dimensionless=True)
        clg_part_load_ratio = create_curve_quadratic(
            model, cool_plf_fplr_coeffs, 'Cool-PLF-fPLR', 0, 1, 0, 1,
            is_dimensionless=True)

    else:  # default curve set, type == 'Split AC' || 'PTAC'
        clg_cap_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_cap_f_of_temp.setCoefficient1Constant(0.942587793)
        clg_cap_f_of_temp.setCoefficient2x(0.009543347)
        clg_cap_f_of_temp.setCoefficient3xPOW2(0.00068377)
        clg_cap_f_of_temp.setCoefficient4y(-0.011042676)
        clg_cap_f_of_temp.setCoefficient5yPOW2(0.000005249)
        clg_cap_f_of_temp.setCoefficient6xTIMESY(-0.00000972)
        clg_cap_f_of_temp.setMinimumValueofx(12.77778)
        clg_cap_f_of_temp.setMaximumValueofx(23.88889)
        clg_cap_f_of_temp.setMinimumValueofy(23.88889)
        clg_cap_f_of_temp.setMaximumValueofy(46.11111)

        clg_cap_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_cap_f_of_flow.setCoefficient1Constant(0.8)
        clg_cap_f_of_flow.setCoefficient2x(0.2)
        clg_cap_f_of_flow.setCoefficient3xPOW2(0)
        clg_cap_f_of_flow.setMinimumValueofx(0.5)
        clg_cap_f_of_flow.setMaximumValueofx(1.5)

        clg_energy_input_ratio_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_energy_input_ratio_f_of_temp.setCoefficient1Constant(0.342414409)
        clg_energy_input_ratio_f_of_temp.setCoefficient2x(0.034885008)
        clg_energy_input_ratio_f_of_temp.setCoefficient3xPOW2(-0.0006237)
        clg_energy_input_ratio_f_of_temp.setCoefficient4y(0.004977216)
        clg_energy_input_ratio_f_of_temp.setCoefficient5yPOW2(0.000437951)
        clg_energy_input_ratio_f_of_temp.setCoefficient6xTIMESY(-0.000728028)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofx(12.77778)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofx(23.88889)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofy(23.88889)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofy(46.11111)

        clg_energy_input_ratio_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_energy_input_ratio_f_of_flow.setCoefficient1Constant(1.1552)
        clg_energy_input_ratio_f_of_flow.setCoefficient2x(-0.1808)
        clg_energy_input_ratio_f_of_flow.setCoefficient3xPOW2(0.0256)
        clg_energy_input_ratio_f_of_flow.setMinimumValueofx(0.5)
        clg_energy_input_ratio_f_of_flow.setMaximumValueofx(1.5)

        clg_part_load_ratio = openstudio_model.CurveQuadratic(model)
        clg_part_load_ratio.setCoefficient1Constant(0.85)
        clg_part_load_ratio.setCoefficient2x(0.15)
        clg_part_load_ratio.setCoefficient3xPOW2(0.0)
        clg_part_load_ratio.setMinimumValueofx(0.0)
        clg_part_load_ratio.setMaximumValueofx(1.0)
        clg_part_load_ratio.setMinimumCurveOutput(0.7)
        clg_part_load_ratio.setMaximumCurveOutput(1.0)

    if clg_cap_f_of_temp is not None:
        clg_coil.setTotalCoolingCapacityFunctionOfTemperatureCurve(clg_cap_f_of_temp)
    if clg_cap_f_of_flow is not None:
        clg_coil.setTotalCoolingCapacityFunctionOfFlowFractionCurve(clg_cap_f_of_flow)
    if clg_energy_input_ratio_f_of_temp is not None:
        clg_coil.setEnergyInputRatioFunctionOfTemperatureCurve(
            clg_energy_input_ratio_f_of_temp)
    if clg_energy_input_ratio_f_of_flow is not None:
        clg_coil.setEnergyInputRatioFunctionOfFlowFractionCurve(
            clg_energy_input_ratio_f_of_flow)
    if clg_part_load_ratio is not None:
        clg_coil.setPartLoadFractionCorrelationCurve(clg_part_load_ratio)

    return clg_coil


def create_coil_cooling_dx_two_speed(
        model, air_loop_node=None, name='2spd DX Clg Coil', schedule=None, type=None):
    """Prototype CoilCoolingDXTwoSpeed object.

    Enters in default curves for coil by type of coil

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or nil in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or nil in which case
            default to always on.
        type: [String] the type of two speed DX coil to reference the correct curve set.
    """
    clg_coil = openstudio_model.CoilCoolingDXTwoSpeed(model)

    # add to air loop if specified
    if air_loop_node is not None:
        clg_coil.addToNode(air_loop_node)

    # set coil name
    name = '2spd DX Clg Coil' if name is None else name
    clg_coil.setName(name)

    # set coil availability schedule
    if schedule is None:  # default always on
        coil_availability_schedule = model.alwaysOnDiscreteSchedule()
    elif isinstance(schedule, str):
        if schedule == 'alwaysOffDiscreteSchedule':
            coil_availability_schedule = model.alwaysOffDiscreteSchedule()
        elif schedule == 'alwaysOnDiscreteSchedule':
            coil_availability_schedule = model.alwaysOnDiscreteSchedule()
        else:
            coil_availability_schedule = model_add_schedule(model, schedule)
    else:  # assume that it is an actual schedule object
        coil_availability_schedule = schedule
    clg_coil.setAvailabilitySchedule(coil_availability_schedule)

    clg_cap_f_of_temp = None
    clg_cap_f_of_flow = None
    clg_energy_input_ratio_f_of_temp = None
    clg_energy_input_ratio_f_of_flow = None
    clg_part_load_ratio = None
    clg_cap_f_of_temp_low_spd = None
    clg_energy_input_ratio_f_of_temp_low_spd = None

    # curve sets
    if type == 'OS default':
        pass  # use OS defaults
    elif type == 'Residential Minisplit HP':
        # Performance curves
        # These coefficients are in SI units
        cool_cap_ft_coeffs_si = [0.7531983499655835, 0.003618193903031667, 0.0,
                                 0.006574385031351544, -6.87181191015432e-05, 0.0]
        cool_eir_ft_coeffs_si = [-0.06376924779982301, -0.0013360593470367282,
                                 1.413060577993827e-05, 0.019433076486584752,
                                 -4.91395947154321e-05, -4.909341249475308e-05]
        cool_cap_fflow_coeffs = [1, 0, 0]
        cool_eir_fflow_coeffs = [1, 0, 0]
        cool_plf_fplr_coeffs = [0.89, 0.11, 0]

        # Make the curves
        clg_cap_f_of_temp = create_curve_biquadratic(
            model, cool_cap_ft_coeffs_si, 'Cool-Cap-fT', 0, 100, 0, 100, None, None)
        clg_cap_f_of_flow = create_curve_quadratic(
            model, cool_cap_fflow_coeffs, 'Cool-Cap-fFF', 0, 2, 0, 2, is_dimensionless=True)
        clg_energy_input_ratio_f_of_temp = create_curve_biquadratic(
            model, cool_eir_ft_coeffs_si, 'Cool-EIR-fT', 0, 100, 0, 100, None, None)
        clg_energy_input_ratio_f_of_flow = create_curve_quadratic(
            model, cool_eir_fflow_coeffs, 'Cool-EIR-fFF', 0, 2, 0, 2, is_dimensionless=True)
        clg_part_load_ratio = create_curve_quadratic(
            model, cool_plf_fplr_coeffs, 'Cool-PLF-fPLR', 0, 1, 0, 1, is_dimensionless=True)
        clg_cap_f_of_temp_low_spd = create_curve_biquadratic(
            model, cool_cap_ft_coeffs_si, 'Cool-Cap-fT', 0, 100, 0, 100, None, None)
        clg_energy_input_ratio_f_of_temp_low_spd = create_curve_biquadratic(
            model, cool_eir_ft_coeffs_si, 'Cool-EIR-fT', 0, 100, 0, 100, None, None)
        clg_coil.setRatedLowSpeedSensibleHeatRatio(0.73)
        clg_coil.setCondenserType('AirCooled')
    else:  # default curve set, type == 'PSZ-AC' || 'Split AC' || 'PTAC'
        clg_cap_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_cap_f_of_temp.setCoefficient1Constant(0.42415)
        clg_cap_f_of_temp.setCoefficient2x(0.04426)
        clg_cap_f_of_temp.setCoefficient3xPOW2(-0.00042)
        clg_cap_f_of_temp.setCoefficient4y(0.00333)
        clg_cap_f_of_temp.setCoefficient5yPOW2(-0.00008)
        clg_cap_f_of_temp.setCoefficient6xTIMESY(-0.00021)
        clg_cap_f_of_temp.setMinimumValueofx(17.0)
        clg_cap_f_of_temp.setMaximumValueofx(22.0)
        clg_cap_f_of_temp.setMinimumValueofy(13.0)
        clg_cap_f_of_temp.setMaximumValueofy(46.0)

        clg_cap_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_cap_f_of_flow.setCoefficient1Constant(0.77136)
        clg_cap_f_of_flow.setCoefficient2x(0.34053)
        clg_cap_f_of_flow.setCoefficient3xPOW2(-0.11088)
        clg_cap_f_of_flow.setMinimumValueofx(0.75918)
        clg_cap_f_of_flow.setMaximumValueofx(1.13877)

        clg_energy_input_ratio_f_of_temp = openstudio_model.CurveBiquadratic(model)
        clg_energy_input_ratio_f_of_temp.setCoefficient1Constant(1.23649)
        clg_energy_input_ratio_f_of_temp.setCoefficient2x(-0.02431)
        clg_energy_input_ratio_f_of_temp.setCoefficient3xPOW2(0.00057)
        clg_energy_input_ratio_f_of_temp.setCoefficient4y(-0.01434)
        clg_energy_input_ratio_f_of_temp.setCoefficient5yPOW2(0.00063)
        clg_energy_input_ratio_f_of_temp.setCoefficient6xTIMESY(-0.00038)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofx(17.0)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofx(22.0)
        clg_energy_input_ratio_f_of_temp.setMinimumValueofy(13.0)
        clg_energy_input_ratio_f_of_temp.setMaximumValueofy(46.0)

        clg_energy_input_ratio_f_of_flow = openstudio_model.CurveQuadratic(model)
        clg_energy_input_ratio_f_of_flow.setCoefficient1Constant(1.20550)
        clg_energy_input_ratio_f_of_flow.setCoefficient2x(-0.32953)
        clg_energy_input_ratio_f_of_flow.setCoefficient3xPOW2(0.12308)
        clg_energy_input_ratio_f_of_flow.setMinimumValueofx(0.75918)
        clg_energy_input_ratio_f_of_flow.setMaximumValueofx(1.13877)

        clg_part_load_ratio = openstudio_model.CurveQuadratic(model)
        clg_part_load_ratio.setCoefficient1Constant(0.77100)
        clg_part_load_ratio.setCoefficient2x(0.22900)
        clg_part_load_ratio.setCoefficient3xPOW2(0.0)
        clg_part_load_ratio.setMinimumValueofx(0.0)
        clg_part_load_ratio.setMaximumValueofx(1.0)

        clg_cap_f_of_temp_low_spd = openstudio_model.CurveBiquadratic(model)
        clg_cap_f_of_temp_low_spd.setCoefficient1Constant(0.42415)
        clg_cap_f_of_temp_low_spd.setCoefficient2x(0.04426)
        clg_cap_f_of_temp_low_spd.setCoefficient3xPOW2(-0.00042)
        clg_cap_f_of_temp_low_spd.setCoefficient4y(0.00333)
        clg_cap_f_of_temp_low_spd.setCoefficient5yPOW2(-0.00008)
        clg_cap_f_of_temp_low_spd.setCoefficient6xTIMESY(-0.00021)
        clg_cap_f_of_temp_low_spd.setMinimumValueofx(17.0)
        clg_cap_f_of_temp_low_spd.setMaximumValueofx(22.0)
        clg_cap_f_of_temp_low_spd.setMinimumValueofy(13.0)
        clg_cap_f_of_temp_low_spd.setMaximumValueofy(46.0)

        clg_energy_input_ratio_f_of_temp_low_spd = openstudio_model.CurveBiquadratic(model)
        clg_energy_input_ratio_f_of_temp_low_spd.setCoefficient1Constant(1.23649)
        clg_energy_input_ratio_f_of_temp_low_spd.setCoefficient2x(-0.02431)
        clg_energy_input_ratio_f_of_temp_low_spd.setCoefficient3xPOW2(0.00057)
        clg_energy_input_ratio_f_of_temp_low_spd.setCoefficient4y(-0.01434)
        clg_energy_input_ratio_f_of_temp_low_spd.setCoefficient5yPOW2(0.00063)
        clg_energy_input_ratio_f_of_temp_low_spd.setCoefficient6xTIMESY(-0.00038)
        clg_energy_input_ratio_f_of_temp_low_spd.setMinimumValueofx(17.0)
        clg_energy_input_ratio_f_of_temp_low_spd.setMaximumValueofx(22.0)
        clg_energy_input_ratio_f_of_temp_low_spd.setMinimumValueofy(13.0)
        clg_energy_input_ratio_f_of_temp_low_spd.setMaximumValueofy(46.0)

        clg_coil.setRatedLowSpeedSensibleHeatRatio(0.69)
        clg_coil.setBasinHeaterCapacity(10)
        clg_coil.setBasinHeaterSetpointTemperature(2.0)

    if clg_cap_f_of_temp is not None:
        clg_coil.setTotalCoolingCapacityFunctionOfTemperatureCurve(clg_cap_f_of_temp)
    if clg_cap_f_of_flow is not None:
        clg_coil.setTotalCoolingCapacityFunctionOfFlowFractionCurve(clg_cap_f_of_flow)
    if clg_energy_input_ratio_f_of_temp is not None:
        clg_coil.setEnergyInputRatioFunctionOfTemperatureCurve(
            clg_energy_input_ratio_f_of_temp)
    if clg_energy_input_ratio_f_of_flow is not None:
        clg_coil.setEnergyInputRatioFunctionOfFlowFractionCurve(
            clg_energy_input_ratio_f_of_flow)
    if clg_part_load_ratio is not None:
        clg_coil.setPartLoadFractionCorrelationCurve(clg_part_load_ratio)
    if clg_cap_f_of_temp_low_spd is not None:
        clg_coil.setLowSpeedTotalCoolingCapacityFunctionOfTemperatureCurve(
            clg_cap_f_of_temp_low_spd)
    if clg_energy_input_ratio_f_of_temp_low_spd is not None:
        clg_coil.setLowSpeedEnergyInputRatioFunctionOfTemperatureCurve(
            clg_energy_input_ratio_f_of_temp_low_spd)

    return clg_coil


def create_coil_cooling_water_to_air_heat_pump_equation_fit(
        model, plant_loop, air_loop_node=None, name='Water-to-Air HP Clg Coil',
        type=None, cop=3.4):
    """Prototype CoilCoolingWaterToAirHeatPumpEquationFit object.

    Enters in default curves for coil by type of coil.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        plant_loop: [<OpenStudio::Model::PlantLoop>] the coil will be placed on
            the demand side of this plant loop.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or nil in which case it will
            be defaulted.
        type: [String] the type of coil to reference the correct curve set.
        cop: [Double] rated cooling coefficient of performance.
    """
    clg_coil = openstudio_model.CoilCoolingWaterToAirHeatPumpEquationFit(model)

    # add to air loop if specified
    if air_loop_node is not None:
        clg_coil.addToNode(air_loop_node)

    # set coil name
    clg_coil.setName(name)

    # add to plant loop
    if plant_loop is None:
        raise ValueError('No plant loop supplied for water cooling coil')
    plant_loop.addDemandBranchForComponent(clg_coil)

    # set coil cop
    cop = 3.4 if cop is None else cop
    clg_coil.setRatedCoolingCoefficientofPerformance(cop)

    # curve sets
    if type == 'OS default':
        pass  # use OS default curves
    else:  # default curve set
        if model.version() < openstudio.VersionString('3.2.0'):
            clg_coil.setTotalCoolingCapacityCoefficient1(-4.30266987344639)
            clg_coil.setTotalCoolingCapacityCoefficient2(7.18536990534372)
            clg_coil.setTotalCoolingCapacityCoefficient3(-2.23946714486189)
            clg_coil.setTotalCoolingCapacityCoefficient4(0.139995928440879)
            clg_coil.setTotalCoolingCapacityCoefficient5(0.102660179888915)
            clg_coil.setSensibleCoolingCapacityCoefficient1(6.0019444814887)
            clg_coil.setSensibleCoolingCapacityCoefficient2(22.6300677244073)
            clg_coil.setSensibleCoolingCapacityCoefficient3(-26.7960783730934)
            clg_coil.setSensibleCoolingCapacityCoefficient4(-1.72374720346819)
            clg_coil.setSensibleCoolingCapacityCoefficient5(0.490644802367817)
            clg_coil.setSensibleCoolingCapacityCoefficient6(0.0693119353468141)
            clg_coil.setCoolingPowerConsumptionCoefficient1(-5.67775976415698)
            clg_coil.setCoolingPowerConsumptionCoefficient2(0.438988156976704)
            clg_coil.setCoolingPowerConsumptionCoefficient3(5.845277342193)
            clg_coil.setCoolingPowerConsumptionCoefficient4(0.141605667000125)
            clg_coil.setCoolingPowerConsumptionCoefficient5(-0.168727936032429)
        else:
            tccc_name = 'Water to Air Heat Pump Total Cooling Capacity Curve'
            if model.getCurveByName(tccc_name).is_initialized():
                total_cooling_capacity_curve = model.getCurveByName(tccc_name).get()
                total_cooling_capacity_curve = \
                    total_cooling_capacity_curve.to_CurveQuadLinear().get()
            else:
                total_cooling_capacity_curve = openstudio_model.CurveQuadLinear(model)
                total_cooling_capacity_curve.setName(tccc_name)
                total_cooling_capacity_curve.setCoefficient1Constant(-4.30266987344639)
                total_cooling_capacity_curve.setCoefficient2w(7.18536990534372)
                total_cooling_capacity_curve.setCoefficient3x(-2.23946714486189)
                total_cooling_capacity_curve.setCoefficient4y(0.139995928440879)
                total_cooling_capacity_curve.setCoefficient5z(0.102660179888915)
                total_cooling_capacity_curve.setMinimumValueofw(-100)
                total_cooling_capacity_curve.setMaximumValueofw(100)
                total_cooling_capacity_curve.setMinimumValueofx(-100)
                total_cooling_capacity_curve.setMaximumValueofx(100)
                total_cooling_capacity_curve.setMinimumValueofy(0)
                total_cooling_capacity_curve.setMaximumValueofy(100)
                total_cooling_capacity_curve.setMinimumValueofz(0)
                total_cooling_capacity_curve.setMaximumValueofz(100)
            clg_coil.setTotalCoolingCapacityCurve(total_cooling_capacity_curve)

            sccc = 'Water to Air Heat Pump Sensible Cooling Capacity Curve'
            if model.getCurveByName(sccc).is_initialized():
                sensible_cooling_capacity_curve = model.getCurveByName(sccc).get()
                sensible_cooling_capacity_curve = \
                    sensible_cooling_capacity_curve.to_CurveQuintLinear().get()
            else:
                sensible_cooling_capacity_curve = openstudio_model.CurveQuintLinear(model)
                sensible_cooling_capacity_curve.setName(sccc)
                sensible_cooling_capacity_curve.setCoefficient1Constant(6.0019444814887)
                sensible_cooling_capacity_curve.setCoefficient2v(22.6300677244073)
                sensible_cooling_capacity_curve.setCoefficient3w(-26.7960783730934)
                sensible_cooling_capacity_curve.setCoefficient4x(-1.72374720346819)
                sensible_cooling_capacity_curve.setCoefficient5y(0.490644802367817)
                sensible_cooling_capacity_curve.setCoefficient6z(0.0693119353468141)
                sensible_cooling_capacity_curve.setMinimumValueofw(-100)
                sensible_cooling_capacity_curve.setMaximumValueofw(100)
                sensible_cooling_capacity_curve.setMinimumValueofx(-100)
                sensible_cooling_capacity_curve.setMaximumValueofx(100)
                sensible_cooling_capacity_curve.setMinimumValueofy(0)
                sensible_cooling_capacity_curve.setMaximumValueofy(100)
                sensible_cooling_capacity_curve.setMinimumValueofz(0)
                sensible_cooling_capacity_curve.setMaximumValueofz(100)
            clg_coil.setSensibleCoolingCapacityCurve(sensible_cooling_capacity_curve)

            cpcc = 'Water to Air Heat Pump Cooling Power Consumption Curve'
            if model.getCurveByName(cpcc).is_initialized():
                cooling_power_consumption_curve = model.getCurveByName(cpcc).get()
                cooling_power_consumption_curve = \
                    cooling_power_consumption_curve.to_CurveQuadLinear().get()
            else:
                cooling_power_consumption_curve = openstudio_model.CurveQuadLinear(model)
                cooling_power_consumption_curve.setName(cpcc)
                cooling_power_consumption_curve.setCoefficient1Constant(-5.67775976415698)
                cooling_power_consumption_curve.setCoefficient2w(0.438988156976704)
                cooling_power_consumption_curve.setCoefficient3x(5.845277342193)
                cooling_power_consumption_curve.setCoefficient4y(0.141605667000125)
                cooling_power_consumption_curve.setCoefficient5z(-0.168727936032429)
                cooling_power_consumption_curve.setMinimumValueofw(-100)
                cooling_power_consumption_curve.setMaximumValueofw(100)
                cooling_power_consumption_curve.setMinimumValueofx(-100)
                cooling_power_consumption_curve.setMaximumValueofx(100)
                cooling_power_consumption_curve.setMinimumValueofy(0)
                cooling_power_consumption_curve.setMaximumValueofy(100)
                cooling_power_consumption_curve.setMinimumValueofz(0)
                cooling_power_consumption_curve.setMaximumValueofz(100)
            clg_coil.setCoolingPowerConsumptionCurve(cooling_power_consumption_curve)

        # part load fraction correlation curve added as a required curve in OS v3.7.0
        if model.version() > openstudio.VersionString('3.6.1'):
            plfcc_name = 'Water to Air Heat Pump Part Load Fraction Correlation Curve'
            if model.getCurveByName(plfcc_name).is_initialized():
                part_load_correlation_curve = model.getCurveByName(plfcc_name).get()
                part_load_correlation_curve = \
                    part_load_correlation_curve.to_CurveLinear().get()
            else:
                part_load_correlation_curve = openstudio_model.CurveLinear(model)
                part_load_correlation_curve.setName(plfcc_name)
                part_load_correlation_curve.setCoefficient1Constant(0.833746458696111)
                part_load_correlation_curve.setCoefficient2x(0.166253541303889)
                part_load_correlation_curve.setMinimumValueofx(0)
                part_load_correlation_curve.setMaximumValueofx(1)
                part_load_correlation_curve.setMinimumCurveOutput(0)
                part_load_correlation_curve.setMaximumCurveOutput(1)
            clg_coil.setPartLoadFractionCorrelationCurve(part_load_correlation_curve)

    return clg_coil
