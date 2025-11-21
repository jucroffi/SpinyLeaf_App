# coding=utf-8
"""Modules taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.CoilHeatingElectric.rb
"""
from __future__ import division

from honeybee_openstudio.openstudio import openstudio, openstudio_model

from .utilities import convert_curve_biquadratic, create_curve_biquadratic, \
    create_curve_quadratic
from .schedule import model_add_schedule


def create_coil_heating_electric(
        model, air_loop_node=None, name='Electric Htg Coil',
        schedule=None, nominal_capacity=None, efficiency=1.0):
    """Prototype CoilHeatingElectric object.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or None in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or None in which
            case default to always on.
        nominal_capacity: [Double] rated nominal capacity.
        efficiency: [Double] rated heating efficiency.
    """
    htg_coil = openstudio_model.CoilHeatingElectric(model)

    # add to air loop if specified
    if air_loop_node is not None:
        htg_coil.addToNode(air_loop_node)

    # set coil name
    htg_coil.setName(name)

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
    htg_coil.setAvailabilitySchedule(coil_availability_schedule)

    # set capacity
    if nominal_capacity is not None:
        htg_coil.setNominalCapacity(nominal_capacity)

    # set efficiency
    if efficiency is not None:
        htg_coil.setEfficiency(efficiency)

    return htg_coil


def create_coil_heating_gas(
        model, air_loop_node=None, name='Gas Htg Coil', schedule=None,
        nominal_capacity=None, efficiency=0.80):
    """Prototype CoilHeatingGas object.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or None in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or None in which
            case default to always on.
        nominal_capacity: [Double] rated nominal capacity.
        efficiency: [Double] rated heating efficiency.
    """
    htg_coil = openstudio_model.CoilHeatingGas(model)

    # add to air loop if specified
    if air_loop_node is not None:
        htg_coil.addToNode(air_loop_node)

    # set coil name
    htg_coil.setName(name)

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
    htg_coil.setAvailabilitySchedule(coil_availability_schedule)

    # set capacity
    if nominal_capacity is not None:
        htg_coil.setNominalCapacity(nominal_capacity)

    # set efficiency
    if efficiency is not None:
        htg_coil.setGasBurnerEfficiency(efficiency)

    # defaults
    if model.version() < openstudio.VersionString('3.7.0'):
        htg_coil.setParasiticElectricLoad(0.0)
        htg_coil.setParasiticGasLoad(0.0)
    else:
        htg_coil.setOnCycleParasiticElectricLoad(0.0)
        htg_coil.setOffCycleParasiticGasLoad(0.0)

    return htg_coil


def create_coil_heating_water(
        model, hot_water_loop, air_loop_node=None, name='Htg Coil', schedule=None,
        rated_inlet_water_temperature=None, rated_outlet_water_temperature=None,
        rated_inlet_air_temperature=16.6, rated_outlet_air_temperature=32.2,
        controller_convergence_tolerance=0.1):
    """Prototype CoilHeatingWater object.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        hot_water_loop: [<OpenStudio::Model::PlantLoop>] the coil will be placed
            on the demand side of this plant loop.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the coil, or nil in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or nil in which case
            default to always on.
        rated_inlet_water_temperature: [Double] rated inlet water temperature in
            degrees Celsius, default is hot water loop design exit temperature.
        rated_outlet_water_temperature: [Double] rated outlet water temperature
            in degrees Celsius, default is hot water loop design return temperature.
        rated_inlet_air_temperature: [Double] rated inlet air temperature in
            degrees Celsius, default is 16.6 (62F).
        rated_outlet_air_temperature: [Double] rated outlet air temperature in
            degrees Celsius, default is 32.2 (90F).
        controller_convergence_tolerance: [Double] controller convergence tolerance.
    """
    htg_coil = openstudio_model.CoilHeatingWater(model)

    # add to hot water loop
    hot_water_loop.addDemandBranchForComponent(htg_coil)

    # add to air loop if specified
    if air_loop_node is not None:
        htg_coil.addToNode(air_loop_node)

    # set coil name
    name = 'Htg Coil' if name is None else name
    htg_coil.setName(name)

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
    htg_coil.setAvailabilitySchedule(coil_availability_schedule)

    # rated water temperatures, use hot water loop temperatures if defined
    if rated_inlet_water_temperature is None:
        rated_inlet_water_temperature = \
            hot_water_loop.sizingPlant().designLoopExitTemperature()
        htg_coil.setRatedInletWaterTemperature(rated_inlet_water_temperature)
    else:
        htg_coil.setRatedInletWaterTemperature(rated_inlet_water_temperature)
    if rated_outlet_water_temperature is None:
        rated_outlet_water_temperature = \
            rated_inlet_water_temperature - \
            hot_water_loop.sizingPlant().loopDesignTemperatureDifference()
        htg_coil.setRatedOutletWaterTemperature(rated_outlet_water_temperature)
    else:
        htg_coil.setRatedOutletWaterTemperature(rated_outlet_water_temperature)

    # rated air temperatures
    if rated_inlet_air_temperature is None:
        htg_coil.setRatedInletAirTemperature(16.6)
    else:
        htg_coil.setRatedInletAirTemperature(rated_inlet_air_temperature)
    if rated_outlet_air_temperature is None:
        htg_coil.setRatedOutletAirTemperature(32.2)
    else:
        htg_coil.setRatedOutletAirTemperature(rated_outlet_air_temperature)

    # coil controller properties
    # note These inputs will get overwritten if addToNode or addDemandBranchForComponent
    # is called on the htg_coil object after this
    htg_coil_controller = htg_coil.controllerWaterCoil().get()
    htg_coil_controller.setName('{} Controller'.format(htg_coil.nameString()))
    htg_coil_controller.setMinimumActuatedFlow(0.0)
    if controller_convergence_tolerance is not None:
        htg_coil_controller.setControllerConvergenceTolerance(
            controller_convergence_tolerance)

    return htg_coil


def create_coil_heating_dx_single_speed(
        model, air_loop_node=None, name='1spd DX Htg Coil', schedule=None, type=None,
        cop=3.3, defrost_strategy='ReverseCycle'):
    """Prototype CoilHeatingDXSingleSpeed object.

    Enters in default curves for coil by type of coil

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or nil in which case it will be defaulted.
        schedule: [String] name of the availability schedule, or
            [<OpenStudio::Model::Schedule>] Schedule object, or nil in which case
            default to always on.
        type: [String] the type of single speed DX coil to reference the correct curve set.
        cop: [Double] rated heating coefficient of performance.
        defrost_strategy: [String] type of defrost strategy. Options are
            reverse-cycle or resistive.
    """
    htg_coil = openstudio_model.CoilHeatingDXSingleSpeed(model)

    # add to air loop if specified
    if air_loop_node is not None:
        htg_coil.addToNode(air_loop_node)

    # set coil name
    htg_coil.setName(name)

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
    htg_coil.setAvailabilitySchedule(coil_availability_schedule)

    # set coil cop
    cop = 3.3 if cop is None else cop
    htg_coil.setRatedCOP(cop)

    htg_cap_f_of_temp = None
    htg_cap_f_of_flow = None
    htg_energy_input_ratio_f_of_temp = None
    htg_energy_input_ratio_f_of_flow = None
    htg_part_load_fraction = None
    def_eir_f_of_temp = None

    # curve sets
    if type == 'OS default':
        pass  # use OS defaults
    elif type == 'Residential Central Air Source HP':
        # Performance curves
        # These coefficients are in IP UNITS
        heat_cap_ft_coeffs_ip = [0.566333415, -0.000744164, -0.0000103,
                                 0.009414634, 0.0000506, -0.00000675]
        heat_eir_ft_coeffs_ip = [0.718398423, 0.003498178, 0.000142202,
                                 -0.005724331, 0.00014085, -0.000215321]
        heat_cap_fflow_coeffs = [0.694045465, 0.474207981, -0.168253446]
        heat_eir_fflow_coeffs = [2.185418751, -1.942827919, 0.757409168]
        heat_plf_fplr_coeffs = [0.8, 0.2, 0]
        defrost_eir_coeffs = [0.1528, 0, 0, 0, 0, 0]

        # Convert coefficients from IP to SI
        heat_cap_ft_coeffs_si = convert_curve_biquadratic(heat_cap_ft_coeffs_ip)
        heat_eir_ft_coeffs_si = convert_curve_biquadratic(heat_eir_ft_coeffs_ip)

        htg_cap_f_of_temp = create_curve_biquadratic(
            model, heat_cap_ft_coeffs_si, 'Heat-Cap-fT', 0, 100, 0, 100, None, None)
        htg_cap_f_of_flow = create_curve_quadratic(
            model, heat_cap_fflow_coeffs, 'Heat-Cap-fFF', 0, 2, 0, 2, is_dimensionless=True)
        htg_energy_input_ratio_f_of_temp = create_curve_biquadratic(
            model, heat_eir_ft_coeffs_si, 'Heat-EIR-fT', 0, 100, 0, 100, None, None)
        htg_energy_input_ratio_f_of_flow = create_curve_quadratic(
            model, heat_eir_fflow_coeffs, 'Heat-EIR-fFF', 0, 2, 0, 2, is_dimensionless=True)
        htg_part_load_fraction = create_curve_quadratic(
            model, heat_plf_fplr_coeffs, 'Heat-PLF-fPLR', 0, 1, 0, 1, is_dimensionless=True)

        # Heating defrost curve for reverse cycle
        def_eir_f_of_temp = create_curve_biquadratic(
            model, defrost_eir_coeffs, 'DefrostEIR', -100, 100, -100, 100, None, None)
    elif type == 'Residential Minisplit HP':
        # Performance curves
        # These coefficients are in SI UNITS
        heat_cap_ft_coeffs_si = [1.14715889038462, -0.010386676170938,
                                 0, 0.00865384615384615, 0, 0]
        heat_eir_ft_coeffs_si = [0.9999941697687026, 0.004684593830254383,
                                 5.901286675833333e-05, -0.0028624467783091973,
                                 1.3041120194135802e-05, -0.00016172918478765433]
        heat_cap_fflow_coeffs = [1, 0, 0]
        heat_eir_fflow_coeffs = [1, 0, 0]
        heat_plf_fplr_coeffs = [0.89, 0.11, 0]
        defrost_eir_coeffs = [0.1528, 0, 0, 0, 0, 0]

        htg_cap_f_of_temp = create_curve_biquadratic(
            model, heat_cap_ft_coeffs_si, 'Heat-Cap-fT', -100, 100, -100, 100, None, None)
        htg_cap_f_of_flow = create_curve_quadratic(
            model, heat_cap_fflow_coeffs, 'Heat-Cap-fFF', 0, 2, 0, 2, is_dimensionless=True)
        htg_energy_input_ratio_f_of_temp = create_curve_biquadratic(
            model, heat_eir_ft_coeffs_si, 'Heat-EIR-fT', -100, 100, -100, 100, None, None)
        htg_energy_input_ratio_f_of_flow = create_curve_quadratic(
            model, heat_eir_fflow_coeffs, 'Heat-EIR-fFF', 0, 2, 0, 2, is_dimensionless=True)
        htg_part_load_fraction = create_curve_quadratic(
            model, heat_plf_fplr_coeffs, 'Heat-PLF-fPLR', 0, 1, 0.6, 1, is_dimensionless=True)

        # Heating defrost curve for reverse cycle
        def_eir_f_of_temp = create_curve_biquadratic(
            model, defrost_eir_coeffs, 'Defrost EIR', -100, 100, -100, 100, None, None)
    else:  # default curve set
        coil_name = htg_coil.nameString()
        htg_cap_f_of_temp = openstudio_model.CurveCubic(model)
        htg_cap_f_of_temp.setName('{} Htg Cap Func of Temp Curve'.format(coil_name))
        htg_cap_f_of_temp.setCoefficient1Constant(0.758746)
        htg_cap_f_of_temp.setCoefficient2x(0.027626)
        htg_cap_f_of_temp.setCoefficient3xPOW2(0.000148716)
        htg_cap_f_of_temp.setCoefficient4xPOW3(0.0000034992)
        htg_cap_f_of_temp.setMinimumValueofx(-20.0)
        htg_cap_f_of_temp.setMaximumValueofx(20.0)

        htg_cap_f_of_flow = openstudio_model.CurveCubic(model)
        htg_cap_f_of_flow.setName('{} Htg Cap Func of Flow Frac Curve'.format(coil_name))
        htg_cap_f_of_flow.setCoefficient1Constant(0.84)
        htg_cap_f_of_flow.setCoefficient2x(0.16)
        htg_cap_f_of_flow.setCoefficient3xPOW2(0.0)
        htg_cap_f_of_flow.setCoefficient4xPOW3(0.0)
        htg_cap_f_of_flow.setMinimumValueofx(0.5)
        htg_cap_f_of_flow.setMaximumValueofx(1.5)

        htg_energy_input_ratio_f_of_temp = openstudio_model.CurveCubic(model)
        htg_energy_input_ratio_f_of_temp.setName(
            '{} EIR Func of Temp Curve'.format(coil_name))
        htg_energy_input_ratio_f_of_temp.setCoefficient1Constant(1.19248)
        htg_energy_input_ratio_f_of_temp.setCoefficient2x(-0.0300438)
        htg_energy_input_ratio_f_of_temp.setCoefficient3xPOW2(0.00103745)
        htg_energy_input_ratio_f_of_temp.setCoefficient4xPOW3(-0.000023328)
        htg_energy_input_ratio_f_of_temp.setMinimumValueofx(-20.0)
        htg_energy_input_ratio_f_of_temp.setMaximumValueofx(20.0)

        htg_energy_input_ratio_f_of_flow = openstudio_model.CurveQuadratic(model)
        htg_energy_input_ratio_f_of_flow.setName(
            '{} EIR Func of Flow Frac Curve'.format(coil_name))
        htg_energy_input_ratio_f_of_flow.setCoefficient1Constant(1.3824)
        htg_energy_input_ratio_f_of_flow.setCoefficient2x(-0.4336)
        htg_energy_input_ratio_f_of_flow.setCoefficient3xPOW2(0.0512)
        htg_energy_input_ratio_f_of_flow.setMinimumValueofx(0.0)
        htg_energy_input_ratio_f_of_flow.setMaximumValueofx(1.0)

        htg_part_load_fraction = openstudio_model.CurveQuadratic(model)
        htg_part_load_fraction.setName('{} PLR Correlation Curve'.format(coil_name))
        htg_part_load_fraction.setCoefficient1Constant(0.85)
        htg_part_load_fraction.setCoefficient2x(0.15)
        htg_part_load_fraction.setCoefficient3xPOW2(0.0)
        htg_part_load_fraction.setMinimumValueofx(0.0)
        htg_part_load_fraction.setMaximumValueofx(1.0)

        if defrost_strategy != 'Resistive':
            def_eir_f_of_temp = openstudio_model.CurveBiquadratic(model)
            def_eir_f_of_temp.setName(
                '{} Defrost EIR Func of Temp Curve'.format(coil_name))
            def_eir_f_of_temp.setCoefficient1Constant(0.297145)
            def_eir_f_of_temp.setCoefficient2x(0.0430933)
            def_eir_f_of_temp.setCoefficient3xPOW2(-0.000748766)
            def_eir_f_of_temp.setCoefficient4y(0.00597727)
            def_eir_f_of_temp.setCoefficient5yPOW2(0.000482112)
            def_eir_f_of_temp.setCoefficient6xTIMESY(-0.000956448)
            def_eir_f_of_temp.setMinimumValueofx(-23.33333)
            def_eir_f_of_temp.setMaximumValueofx(29.44444)
            def_eir_f_of_temp.setMinimumValueofy(-23.33333)
            def_eir_f_of_temp.setMaximumValueofy(29.44444)

    if type == 'PSZ-AC':
        htg_coil.setMinimumOutdoorDryBulbTemperatureforCompressorOperation(-12.2)
        htg_coil.setMaximumOutdoorDryBulbTemperatureforDefrostOperation(1.67)
        htg_coil.setCrankcaseHeaterCapacity(50.0)
        htg_coil.setMaximumOutdoorDryBulbTemperatureforCrankcaseHeaterOperation(4.4)
        htg_coil.setDefrostControl('OnDemand')

        def_eir_f_of_temp = openstudio_model.CurveBiquadratic(model)
        def_eir_f_of_temp.setName('{} Defrost EIR Func of Temp Curve'.format(coil_name))
        def_eir_f_of_temp.setCoefficient1Constant(0.297145)
        def_eir_f_of_temp.setCoefficient2x(0.0430933)
        def_eir_f_of_temp.setCoefficient3xPOW2(-0.000748766)
        def_eir_f_of_temp.setCoefficient4y(0.00597727)
        def_eir_f_of_temp.setCoefficient5yPOW2(0.000482112)
        def_eir_f_of_temp.setCoefficient6xTIMESY(-0.000956448)
        def_eir_f_of_temp.setMinimumValueofx(-23.33333)
        def_eir_f_of_temp.setMaximumValueofx(29.44444)
        def_eir_f_of_temp.setMinimumValueofy(-23.33333)
        def_eir_f_of_temp.setMaximumValueofy(29.44444)

    if htg_cap_f_of_temp is not None:
        htg_coil.setTotalHeatingCapacityFunctionofTemperatureCurve(htg_cap_f_of_temp)
    if htg_cap_f_of_flow is not None:
        htg_coil.setTotalHeatingCapacityFunctionofFlowFractionCurve(htg_cap_f_of_flow)
    if htg_energy_input_ratio_f_of_temp is not None:
        htg_coil.setEnergyInputRatioFunctionofTemperatureCurve(
            htg_energy_input_ratio_f_of_temp)
    if htg_energy_input_ratio_f_of_flow is not None:
        htg_coil.setEnergyInputRatioFunctionofFlowFractionCurve(
            htg_energy_input_ratio_f_of_flow)
    if htg_part_load_fraction is not None:
        htg_coil.setPartLoadFractionCorrelationCurve(htg_part_load_fraction)
    if def_eir_f_of_temp is not None:
        htg_coil.setDefrostEnergyInputRatioFunctionofTemperatureCurve(def_eir_f_of_temp)
    htg_coil.setDefrostStrategy(defrost_strategy)
    htg_coil.setDefrostControl('OnDemand')

    return htg_coil


def create_coil_heating_water_to_air_heat_pump_equation_fit(
        model, plant_loop, air_loop_node=None,
        name='Water-to-Air HP Htg Coil', type=None, cop=4.2):
    """Prototype CoilHeatingWaterToAirHeatPumpEquationFit object.

    Enters in default curves for coil by type of coil.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        plant_loop: [<OpenStudio::Model::PlantLoop>] the coil will be placed on
            the demand side of this plant loop.
        air_loop_node: [<OpenStudio::Model::Node>] the coil will be placed on
            this node of the air loop.
        name: [String] the name of the system, or None in which case it will
            be defaulted.
        type: [String] the type of coil to reference the correct curve set.
        cop: [Double] rated heating coefficient of performance.
    """
    htg_coil = openstudio_model.CoilHeatingWaterToAirHeatPumpEquationFit(model)

    # add to air loop if specified
    if air_loop_node is not None:
        htg_coil.addToNode(air_loop_node)

    # set coil name
    htg_coil.setName(name)

    # add to plant loop
    if plant_loop is None:
        raise ValueError('No plant loop supplied for heating coil')
    plant_loop.addDemandBranchForComponent(htg_coil)

    # set coil cop
    cop = 4.2 if cop is None else cop
    htg_coil.setRatedHeatingCoefficientofPerformance(cop)

    # curve sets
    if type == 'OS default':
        pass  # use OS default curves
    else:  # default curve set
        if model.version() < openstudio.VersionString('3.2.0'):
            htg_coil.setHeatingCapacityCoefficient1(0.237847462869254)
            htg_coil.setHeatingCapacityCoefficient2(-3.35823796081626)
            htg_coil.setHeatingCapacityCoefficient3(3.80640467406376)
            htg_coil.setHeatingCapacityCoefficient4(0.179200417311554)
            htg_coil.setHeatingCapacityCoefficient5(0.12860719846082)
            htg_coil.setHeatingPowerConsumptionCoefficient1(-3.79175529243238)
            htg_coil.setHeatingPowerConsumptionCoefficient2(3.38799239505527)
            htg_coil.setHeatingPowerConsumptionCoefficient3(1.5022612076303)
            htg_coil.setHeatingPowerConsumptionCoefficient4(-0.177653510577989)
            htg_coil.setHeatingPowerConsumptionCoefficient5(-0.103079864171839)
        else:
            hcc_name = 'Water to Air Heat Pump Heating Capacity Curve'
            hcc = model.getCurveByName(hcc_name)
            if hcc.is_initialized():
                heating_capacity_curve = hcc.get()
                heating_capacity_curve = heating_capacity_curve.to_CurveQuadLinear().get()
            else:
                heating_capacity_curve = openstudio_model.CurveQuadLinear(model)
                heating_capacity_curve.setName(hcc_name)
                heating_capacity_curve.setCoefficient1Constant(0.237847462869254)
                heating_capacity_curve.setCoefficient2w(-3.35823796081626)
                heating_capacity_curve.setCoefficient3x(3.80640467406376)
                heating_capacity_curve.setCoefficient4y(0.179200417311554)
                heating_capacity_curve.setCoefficient5z(0.12860719846082)
                heating_capacity_curve.setMinimumValueofw(-100)
                heating_capacity_curve.setMaximumValueofw(100)
                heating_capacity_curve.setMinimumValueofx(-100)
                heating_capacity_curve.setMaximumValueofx(100)
                heating_capacity_curve.setMinimumValueofy(0)
                heating_capacity_curve.setMaximumValueofy(100)
                heating_capacity_curve.setMinimumValueofz(0)
                heating_capacity_curve.setMaximumValueofz(100)
            htg_coil.setHeatingCapacityCurve(heating_capacity_curve)

            pcc_name = 'Water to Air Heat Pump Heating Power Consumption Curve'
            pcc = model.getCurveByName(pcc_name)
            if pcc.is_initialized():
                heating_power_consumption_curve = pcc.get()
                heating_power_consumption_curve = \
                    heating_power_consumption_curve.to_CurveQuadLinear().get()
            else:
                heating_power_consumption_curve = openstudio_model.CurveQuadLinear(model)
                heating_power_consumption_curve.setName(pcc_name)
                heating_power_consumption_curve.setCoefficient1Constant(-3.79175529243238)
                heating_power_consumption_curve.setCoefficient2w(3.38799239505527)
                heating_power_consumption_curve.setCoefficient3x(1.5022612076303)
                heating_power_consumption_curve.setCoefficient4y(-0.177653510577989)
                heating_power_consumption_curve.setCoefficient5z(-0.103079864171839)
                heating_power_consumption_curve.setMinimumValueofw(-100)
                heating_power_consumption_curve.setMaximumValueofw(100)
                heating_power_consumption_curve.setMinimumValueofx(-100)
                heating_power_consumption_curve.setMaximumValueofx(100)
                heating_power_consumption_curve.setMinimumValueofy(0)
                heating_power_consumption_curve.setMaximumValueofy(100)
                heating_power_consumption_curve.setMinimumValueofz(0)
                heating_power_consumption_curve.setMaximumValueofz(100)
            htg_coil.setHeatingPowerConsumptionCurve(heating_power_consumption_curve)

        # part load fraction correlation curve added as a required curve in OS v3.7.0
        if model.version() > openstudio.VersionString('3.6.1'):
            plfcc_name = 'Water to Air Heat Pump Part Load Fraction Correlation Curve'
            plfcc = model.getCurveByName(plfcc_name)
            if plfcc.is_initialized():
                part_load_correlation_curve = plfcc.get()
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
            htg_coil.setPartLoadFractionCorrelationCurve(part_load_correlation_curve)

    return htg_coil
