# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.hvac_systems.rb
"""
from __future__ import division
import sys

from ladybug.datatype.temperature import Temperature
from ladybug.datatype.temperaturedelta import TemperatureDelta
from ladybug.datatype.pressure import Pressure
from ladybug.datatype.volumeflowrate import VolumeFlowRate
from ladybug.datatype.power import Power
from ladybug.datatype.distance import Distance

from honeybee_openstudio.openstudio import openstudio, openstudio_model, \
    os_vector_len, os_create_vector
from .utilities import kw_per_ton_to_cop, eer_to_cop_no_fan, hspf_to_cop_no_fan, \
    ems_friendly_name, rename_plant_loop_nodes
from .schedule import create_constant_schedule_ruleset, model_add_schedule
from .thermal_zone import thermal_zone_get_outdoor_airflow_rate, \
    thermal_zone_get_outdoor_airflow_rate_per_area

from .central_air_source_heat_pump import create_central_air_source_heat_pump
from .boiler_hot_water import create_boiler_hot_water
from .plant_loop import chw_sizing_control, plant_loop_set_chw_pri_sec_configuration
from .pump_variable_speed import pump_variable_speed_set_control_type
from .cooling_tower import prototype_apply_condenser_water_temperatures
from .fan import create_fan_by_name, fan_change_impeller_efficiency
from .coil_heating import create_coil_heating_electric, create_coil_heating_gas, \
    create_coil_heating_water, create_coil_heating_dx_single_speed, \
    create_coil_heating_water_to_air_heat_pump_equation_fit
from .coil_cooling import create_coil_cooling_water, create_coil_cooling_dx_single_speed, \
    create_coil_cooling_water_to_air_heat_pump_equation_fit, \
    create_coil_cooling_dx_two_speed
from .heat_recovery import create_hx_air_to_air_sensible_and_latent
from .sizing_system import adjust_sizing_system
from .air_conditioner_variable_refrigerant_flow import \
    create_air_conditioner_variable_refrigerant_flow
from .radiant_system_controls import model_add_radiant_proportional_controls, \
    model_add_radiant_basic_controls

TEMPERATURE = Temperature()
TEMP_DELTA = TemperatureDelta()
PRESSURE = Pressure()
FLOW_RATE = VolumeFlowRate()
POWER = Power()
DISTANCE = Distance()


def standard_design_sizing_temperatures():
    """Get a dictionary of design sizing temperatures for lookups."""
    dsgn_temps = {}
    dsgn_temps['prehtg_dsgn_sup_air_temp_f'] = 45.0
    dsgn_temps['preclg_dsgn_sup_air_temp_f'] = 55.0
    dsgn_temps['htg_dsgn_sup_air_temp_f'] = 55.0
    dsgn_temps['clg_dsgn_sup_air_temp_f'] = 55.0
    dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 104.0
    dsgn_temps['zn_clg_dsgn_sup_air_temp_f'] = 55.0
    dsgn_temps_c = {}
    for key, val in dsgn_temps.items():
        dsgn_temps_c['{}_c'.format(key[:-2])] = TEMPERATURE.to_unit([val], 'C', 'F')[0]
    dsgn_temps.update(dsgn_temps_c)
    return dsgn_temps


def model_add_hw_loop(
        model, boiler_fuel_type, ambient_loop=None, system_name='Hot Water Loop',
        dsgn_sup_wtr_temp=180.0, dsgn_sup_wtr_temp_delt=20.0, pump_spd_ctrl='Variable',
        pump_tot_hd=None, boiler_draft_type=None, boiler_eff_curve_temp_eval_var=None,
        boiler_lvg_temp_dsgn=None, boiler_out_temp_lmt=None,
        boiler_max_plr=None, boiler_sizing_factor=None):
    """Create a hot water loop with a boiler, district heat, or water-to-water heat pump.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object
        boiler_fuel_type: [String] valid choices are Electricity, NaturalGas,
            Propane, PropaneGas, FuelOilNo1, FuelOilNo2, DistrictHeating,
            DistrictHeatingWater, DistrictHeatingSteam, HeatPump
        ambient_loop: [OpenStudio::Model::PlantLoop] The condenser loop for the
            heat pump. Only used when boiler_fuel_type is HeatPump.
        system_name: [String] the name of the system. If None, it will be defaulted.
        dsgn_sup_wtr_temp: [Double] design supply water temperature in degrees
            Fahrenheit, default 180F.
        dsgn_sup_wtr_temp_delt: [Double] design supply-return water temperature
            difference in degrees Rankine, default 20R.
        pump_spd_ctrl: [String] pump speed control type, Constant or Variable (default).
        pump_tot_hd: [Double] pump head in ft H2O.
        boiler_draft_type: [String] Boiler type Condensing, MechanicalNoncondensing,
            Natural (default).
        boiler_eff_curve_temp_eval_var: [String] LeavingBoiler or EnteringBoiler
            temperature for the boiler efficiency curve.
        boiler_lvg_temp_dsgn: [Double] boiler leaving design temperature in
            degrees Fahrenheit.
        boiler_out_temp_lmt: [Double] boiler outlet temperature limit in
            degrees Fahrenheit.
        boiler_max_plr: [Double] boiler maximum part load ratio.
        boiler_sizing_factor: [Double] boiler oversizing factor.

    Returns:
        [OpenStudio::Model::PlantLoop] the resulting hot water loop.
    """
    # create hot water loop
    hot_water_loop = openstudio_model.PlantLoop(model)
    if system_name is None:
        hot_water_loop.setName('Hot Water Loop')
    else:
        hot_water_loop.setName(system_name)

    # hot water loop sizing and controls
    dsgn_sup_wtr_temp = 180.0 if dsgn_sup_wtr_temp is None else dsgn_sup_wtr_temp
    dsgn_sup_wtr_temp_c = TEMPERATURE.to_unit([dsgn_sup_wtr_temp], 'C', 'F')[0]
    dsgn_sup_wtr_temp_delt = 20.0 if dsgn_sup_wtr_temp_delt is None \
        else dsgn_sup_wtr_temp_delt
    dsgn_sup_wtr_temp_delt_k = TEMP_DELTA.to_unit([dsgn_sup_wtr_temp_delt], 'dC', 'dF')[0]

    sizing_plant = hot_water_loop.sizingPlant()
    sizing_plant.setLoopType('Heating')
    sizing_plant.setDesignLoopExitTemperature(dsgn_sup_wtr_temp_c)
    sizing_plant.setLoopDesignTemperatureDifference(dsgn_sup_wtr_temp_delt_k)
    hot_water_loop.setMinimumLoopTemperature(10.0)
    hw_temp_sch = create_constant_schedule_ruleset(
        model, dsgn_sup_wtr_temp_c,
        name='{} Temp - {}F'.format(hot_water_loop.nameString(), int(dsgn_sup_wtr_temp)),
        schedule_type_limit='Temperature')
    hw_stpt_manager = openstudio_model.SetpointManagerScheduled(model, hw_temp_sch)
    hw_stpt_manager.setName('{} Setpoint Manager'.format(hot_water_loop.nameString()))
    hw_stpt_manager.addToNode(hot_water_loop.supplyOutletNode())

    # create hot water pump
    if pump_spd_ctrl == 'Constant':
        hw_pump = openstudio_model.PumpConstantSpeed(model)
    elif pump_spd_ctrl == 'Variable':
        hw_pump = openstudio_model.PumpVariableSpeed(model)
    else:
        hw_pump = openstudio_model.PumpVariableSpeed(model)
    hw_pump.setName('{} Pump'.format(hot_water_loop.nameString()))
    if pump_tot_hd is None:
        pump_tot_hd_pa = PRESSURE.to_unit([60 * 12], 'Pa', 'inH2O')[0]
    else:
        pump_tot_hd_pa = PRESSURE.to_unit([pump_tot_hd * 12], 'Pa', 'inH2O')[0]
    hw_pump.setRatedPumpHead(pump_tot_hd_pa)
    hw_pump.setMotorEfficiency(0.9)
    hw_pump.setPumpControlType('Intermittent')
    hw_pump.addToNode(hot_water_loop.supplyInletNode())

    # switch statement to handle district heating name change
    if model.version() < openstudio.VersionString('3.7.0'):
        if boiler_fuel_type == 'DistrictHeatingWater' or \
                boiler_fuel_type == 'DistrictHeatingSteam':
            boiler_fuel_type = 'DistrictHeating'
    else:
        if boiler_fuel_type == 'DistrictHeating':
            boiler_fuel_type = 'DistrictHeatingWater'

    # create boiler and add to loop
    # District Heating
    if boiler_fuel_type == 'DistrictHeating':
        district_heat = openstudio_model.DistrictHeating(model)
        district_heat.setName('{} District Heating'.format(hot_water_loop.nameString()))
        district_heat.autosizeNominalCapacity()
        hot_water_loop.addSupplyBranchForComponent(district_heat)
    elif boiler_fuel_type == 'DistrictHeatingWater':
        district_heat = openstudio_model.DistrictHeatingWater(model)
        district_heat.setName('{} District Heating'.format(hot_water_loop.nameString()))
        district_heat.autosizeNominalCapacity()
        hot_water_loop.addSupplyBranchForComponent(district_heat)
    elif boiler_fuel_type == 'DistrictHeatingSteam':
        district_heat = openstudio_model.DistrictHeatingSteam(model)
        district_heat.setName('{} District Heating'.format(hot_water_loop.nameString()))
        district_heat.autosizeNominalCapacity()
        hot_water_loop.addSupplyBranchForComponent(district_heat)
    elif boiler_fuel_type in ('HeatPump', 'AmbientLoop'):
        # Ambient Loop
        water_to_water_hp = openstudio_model.HeatPumpWaterToWaterEquationFitHeating(model)
        water_to_water_hp.setName(
            '{} Water to Water Heat Pump'.format(hot_water_loop.nameString()))
        hot_water_loop.addSupplyBranchForComponent(water_to_water_hp)
        # Get or add an ambient loop
        if ambient_loop is None:
            ambient_loop = model_get_or_add_ambient_water_loop(model)
        ambient_loop.addDemandBranchForComponent(water_to_water_hp)
    elif boiler_fuel_type in ('AirSourceHeatPump', 'ASHP'):
        # Central Air Source Heat Pump
        create_central_air_source_heat_pump(model, hot_water_loop)
    elif boiler_fuel_type in ('Electricity', 'Gas', 'NaturalGas', 'Propane',
                              'PropaneGas', 'FuelOilNo1', 'FuelOilNo2'):
        # Boiler
        lvg_temp_dsgn_f = dsgn_sup_wtr_temp if boiler_lvg_temp_dsgn is None \
            else boiler_lvg_temp_dsgn
        out_temp_lmt_f = 203.0 if boiler_out_temp_lmt is None else boiler_out_temp_lmt
        create_boiler_hot_water(
            model, hot_water_loop=hot_water_loop, fuel_type=boiler_fuel_type,
            draft_type=boiler_draft_type, nominal_thermal_efficiency=0.78,
            eff_curve_temp_eval_var=boiler_eff_curve_temp_eval_var,
            lvg_temp_dsgn_f=lvg_temp_dsgn_f, out_temp_lmt_f=out_temp_lmt_f,
            max_plr=boiler_max_plr, sizing_factor=boiler_sizing_factor)
    else:
        msg = 'Boiler fuel type {} is not valid, no boiler will be added.'.format(
            boiler_fuel_type)
        print(msg)

    # add hot water loop pipes
    supply_equipment_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    supply_equipment_bypass_pipe.setName(
        '{} Supply Equipment Bypass'.format(hot_water_loop.nameString()))
    hot_water_loop.addSupplyBranchForComponent(supply_equipment_bypass_pipe)

    coil_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    coil_bypass_pipe.setName('{} Coil Bypass'.format(hot_water_loop.nameString()))
    hot_water_loop.addDemandBranchForComponent(coil_bypass_pipe)

    supply_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    supply_outlet_pipe.setName('{} Supply Outlet'.format(hot_water_loop.nameString()))
    supply_outlet_pipe.addToNode(hot_water_loop.supplyOutletNode())

    demand_inlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_inlet_pipe.setName('{} Demand Inlet'.format(hot_water_loop.nameString()))
    demand_inlet_pipe.addToNode(hot_water_loop.demandInletNode())

    demand_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_outlet_pipe.setName('{} Demand Outlet'.format(hot_water_loop.nameString()))
    demand_outlet_pipe.addToNode(hot_water_loop.demandOutletNode())

    return hot_water_loop


def model_add_chw_loop(
        model, system_name='Chilled Water Loop', cooling_fuel='Electricity',
        dsgn_sup_wtr_temp=44.0, dsgn_sup_wtr_temp_delt=10.1, chw_pumping_type=None,
        chiller_cooling_type=None, chiller_condenser_type=None,
        chiller_compressor_type=None, num_chillers=1,
        condenser_water_loop=None, waterside_economizer='none'):
    """Create a chilled water loop and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        system_name: [String] the name of the system, or None in which case
            it will be defaulted.
        cooling_fuel: [String] cooling fuel. Valid choices are: Electricity,
            DistrictCooling.
        dsgn_sup_wtr_temp: [Double] design supply water temperature in degrees
            Fahrenheit, default 44F.
        dsgn_sup_wtr_temp_delt: [Double] design supply-return water temperature
            difference in degrees Rankine, default 10 dF.
        chw_pumping_type: [String] valid choices are const_pri, const_pri_var_sec.
        chiller_cooling_type: [String] valid choices are AirCooled, WaterCooled.
        chiller_condenser_type: [String] valid choices are WithCondenser,
            WithoutCondenser, None.
        chiller_compressor_type: [String] valid choices are Centrifugal,
            Reciprocating, Rotary Screw, Scroll, None.
        num_chillers: [Integer] the number of chillers.
        condenser_water_loop: [OpenStudio::Model::PlantLoop] optional condenser
            water loop for water-cooled chillers. If None, the chillers will
            be air cooled.
        waterside_economizer: [String] Options are none, integrated, non-integrated.
            If integrated will add a heat exchanger to the supply inlet of the
            chilled water loop to provide waterside economizing whenever wet
            bulb temperatures allow. Non-integrated will add a heat exchanger
            in parallel with the chiller that will operate only when it can
            meet cooling demand exclusively with the waterside economizing.

    Returns:
        [OpenStudio::Model::PlantLoop] the resulting chilled water loop.
    """
    # create chilled water loop
    chilled_water_loop = openstudio_model.PlantLoop(model)
    if system_name is None:
        chilled_water_loop.setName('Chilled Water Loop')
    else:
        chilled_water_loop.setName(system_name)
    dsgn_sup_wtr_temp = 44 if dsgn_sup_wtr_temp is None else dsgn_sup_wtr_temp

    # chilled water loop sizing and controls
    chw_sizing_control(model, chilled_water_loop, dsgn_sup_wtr_temp, dsgn_sup_wtr_temp_delt)

    # create chilled water pumps
    if chw_pumping_type == 'const_pri':
        # primary chilled water pump
        pri_chw_pump = openstudio_model.PumpVariableSpeed(model)
        pri_chw_pump.setName('{} Pump'.format(chilled_water_loop.nameString()))
        pri_chw_pump.setRatedPumpHead(PRESSURE.to_unit([60 * 12], 'Pa', 'inH2O')[0])
        pri_chw_pump.setMotorEfficiency(0.9)
        # flat pump curve makes it behave as a constant speed pump
        pri_chw_pump.setFractionofMotorInefficienciestoFluidStream(0)
        pri_chw_pump.setCoefficient1ofthePartLoadPerformanceCurve(0)
        pri_chw_pump.setCoefficient2ofthePartLoadPerformanceCurve(1)
        pri_chw_pump.setCoefficient3ofthePartLoadPerformanceCurve(0)
        pri_chw_pump.setCoefficient4ofthePartLoadPerformanceCurve(0)
        pri_chw_pump.setPumpControlType('Intermittent')
        pri_chw_pump.addToNode(chilled_water_loop.supplyInletNode())
    elif chw_pumping_type == 'const_pri_var_sec':
        pri_sec_config = plant_loop_set_chw_pri_sec_configuration(model)

        if pri_sec_config == 'common_pipe':
            # primary chilled water pump
            pri_chw_pump = openstudio_model.PumpConstantSpeed(model)
            pri_chw_pump.setName('{} Primary Pump'.format(chilled_water_loop.nameString()))
            pri_chw_pump.setRatedPumpHead(PRESSURE.to_unit([15 * 12], 'Pa', 'inH2O')[0])
            pri_chw_pump.setMotorEfficiency(0.9)
            pri_chw_pump.setPumpControlType('Intermittent')
            pri_chw_pump.addToNode(chilled_water_loop.supplyInletNode())
            # secondary chilled water pump
            sec_chw_pump = openstudio_model.PumpVariableSpeed(model)
            sec_chw_pump.setName('{} Secondary Pump'.format(chilled_water_loop.nameString()))
            sec_chw_pump.setRatedPumpHead(PRESSURE.to_unit([45 * 12], 'Pa', 'inH2O')[0])
            sec_chw_pump.setMotorEfficiency(0.9)
            # curve makes it perform like variable speed pump
            sec_chw_pump.setFractionofMotorInefficienciestoFluidStream(0)
            sec_chw_pump.setCoefficient1ofthePartLoadPerformanceCurve(0)
            sec_chw_pump.setCoefficient2ofthePartLoadPerformanceCurve(0.0205)
            sec_chw_pump.setCoefficient3ofthePartLoadPerformanceCurve(0.4101)
            sec_chw_pump.setCoefficient4ofthePartLoadPerformanceCurve(0.5753)
            sec_chw_pump.setPumpControlType('Intermittent')
            sec_chw_pump.addToNode(chilled_water_loop.demandInletNode())
            # Change the chilled water loop to have a two-way common pipes
            chilled_water_loop.setCommonPipeSimulation('CommonPipe')
        elif pri_sec_config == 'heat_exchanger':
            # Check number of chillers
            if num_chillers > 3:
                msg = 'EMS Code for multiple chiller pump has not been written for ' \
                    'greater than 3 chillers. This has {} chillers'.format(num_chillers)
                print(msg)
            # NOTE: PRECONDITIONING for `const_pri_var_sec` pump type is only applicable
            # for PRM routine and only applies to System Type 7 and System Type 8
            # See: model_add_prm_baseline_system under Model object.
            # In this scenario, we will need to create a primary and secondary configuration:
            # chilled_water_loop is the primary loop
            # Primary: demand: heat exchanger, supply: chillers, name: Chilled Water Loop_Primary
            # Secondary: demand: Coils, supply: heat exchanger, name: Chilled Water Loop
            secondary_chilled_water_loop = openstudio_model.PlantLoop(model)
            secondary_loop_name = 'Chilled Water Loop' if system_name is None \
                else system_name
            # Reset primary loop name
            chilled_water_loop.setName('{}_Primary'.format(secondary_loop_name))
            secondary_chilled_water_loop.setName(secondary_loop_name)
            chw_sizing_control(model, secondary_chilled_water_loop,
                               dsgn_sup_wtr_temp, dsgn_sup_wtr_temp_delt)
            chilled_water_loop.additionalProperties.setFeature('is_primary_loop', True)
            chilled_water_loop.additionalProperties.setFeature(
                'secondary_loop_name', secondary_chilled_water_loop.nameString())
            secondary_chilled_water_loop.additionalProperties.setFeature(
                'is_secondary_loop', True)
            # primary chilled water pumps are added when adding chillers
            # Add Constant pump, in plant loop, the number of chiller adjustment
            # will assign pump to each chiller
            pri_chw_pump = openstudio_model.PumpVariableSpeed(model)
            pump_variable_speed_set_control_type(
                pri_chw_pump, control_type='Riding Curve')
            # This pump name is important for function
            # add_ems_for_multiple_chiller_pumps_w_secondary_plant
            # If you update it here, you must update the logic there to account for this
            pri_chw_pump.setName('{} Primary Pump'.format(chilled_water_loop.nameString()))
            # Will need to adjust the pump power after a sizing run
            pri_chw_pump.setRatedPumpHead(
                PRESSURE.to_unit([15 * 12], 'Pa', 'inH2O')[0] / num_chillers)
            pri_chw_pump.setMotorEfficiency(0.9)
            pri_chw_pump.setPumpControlType('Intermittent')
            pri_chw_pump.addToNode(chilled_water_loop.supplyInletNode())

            # secondary chilled water pump
            sec_chw_pump = openstudio_model.PumpVariableSpeed(model)
            sec_chw_pump.setName('{} Pump'.format(secondary_chilled_water_loop.nameString()))
            sec_chw_pump.setRatedPumpHead(PRESSURE.to_unit([45 * 12], 'Pa', 'inH2O')[0])
            sec_chw_pump.setMotorEfficiency(0.9)
            # curve makes it perform like variable speed pump
            sec_chw_pump.setFractionofMotorInefficienciestoFluidStream(0)
            sec_chw_pump.setCoefficient1ofthePartLoadPerformanceCurve(0)
            sec_chw_pump.setCoefficient2ofthePartLoadPerformanceCurve(0.0205)
            sec_chw_pump.setCoefficient3ofthePartLoadPerformanceCurve(0.4101)
            sec_chw_pump.setCoefficient4ofthePartLoadPerformanceCurve(0.5753)
            sec_chw_pump.setPumpControlType('Intermittent')
            sec_chw_pump.addToNode(secondary_chilled_water_loop.demandInletNode())

            # Add HX to connect secondary and primary loop
            heat_exchanger = openstudio_model.HeatExchangerFluidToFluid(model)
            secondary_chilled_water_loop.addSupplyBranchForComponent(heat_exchanger)
            chilled_water_loop.addDemandBranchForComponent(heat_exchanger)

            # Clean up connections
            hx_bypass_pipe = openstudio_model.PipeAdiabatic(model)
            hx_bypass_pipe.setName(
                '{} HX Bypass'.format(secondary_chilled_water_loop.nameString()))
            secondary_chilled_water_loop.addSupplyBranchForComponent(hx_bypass_pipe)
            outlet_pipe = openstudio_model.PipeAdiabatic(model)
            outlet_pipe.setName(
                '{} Supply Outlet'.format(secondary_chilled_water_loop.nameString()))
            outlet_pipe.addToNode(secondary_chilled_water_loop.supplyOutletNode())
        else:
            msg = 'No primary/secondary configuration specified for chilled water loop.'
            print(msg)
    else:
        print('No pumping type specified for the chilled water loop.')

    # check for existence of condenser_water_loop if WaterCooled
    if chiller_cooling_type == 'WaterCooled' and condenser_water_loop is None:
        print('Requested chiller is WaterCooled but no condenser loop specified.')

    # check for non-existence of condenser_water_loop if AirCooled
    if chiller_cooling_type == 'AirCooled' and condenser_water_loop is not None:
        print('Requested chiller is AirCooled but condenser loop specified.')

    if cooling_fuel == 'DistrictCooling':
        # DistrictCooling
        dist_clg = openstudio_model.DistrictCooling(model)
        dist_clg.setName('Purchased Cooling')
        dist_clg.autosizeNominalCapacity()
        chilled_water_loop.addSupplyBranchForComponent(dist_clg)
    else:
        # use default efficiency from 90.1-2019
        # 1.188 kw/ton for a 150 ton AirCooled chiller
        # 0.66 kw/ton for a 150 ton Water Cooled positive displacement chiller
        if chiller_cooling_type == 'AirCooled':
            default_cop = kw_per_ton_to_cop(1.188)
        elif chiller_cooling_type == 'WaterCooled':
            default_cop = kw_per_ton_to_cop(0.66)
        else:
            default_cop = kw_per_ton_to_cop(0.66)

        # make the correct type of chiller based these properties
        chiller_sizing_factor = round(1.0 / num_chillers, 2)

        # Create chillers and set plant operation scheme
        for i in range(num_chillers):
            chiller = openstudio_model.ChillerElectricEIR(model)
            cool_t = chiller_cooling_type \
                if chiller_cooling_type is not None else 'WaterCooled'
            cond_t = chiller_condenser_type if chiller_condenser_type is not None else ''
            comp_t = chiller_compressor_type if chiller_compressor_type is not None else ''
            ch_name = 'ASHRAE 90.1 {} {} {} Chiller {}'.format(cool_t, cond_t, comp_t, i)
            chiller.setName(ch_name)
            chilled_water_loop.addSupplyBranchForComponent(chiller)
            dsgn_sup_wtr_temp_c = TEMPERATURE.to_unit([dsgn_sup_wtr_temp], 'C', 'F')[0]
            chiller.setReferenceLeavingChilledWaterTemperature(dsgn_sup_wtr_temp_c)
            lcw_ltl = TEMPERATURE.to_unit([36.0], 'C', 'F')[0]
            chiller.setLeavingChilledWaterLowerTemperatureLimit(lcw_ltl)
            rec_ft = TEMPERATURE.to_unit([95.0], 'C', 'F')[0]
            chiller.setReferenceEnteringCondenserFluidTemperature(rec_ft)
            chiller.setMinimumPartLoadRatio(0.15)
            chiller.setMaximumPartLoadRatio(1.0)
            chiller.setOptimumPartLoadRatio(1.0)
            chiller.setMinimumUnloadingRatio(0.25)
            chiller.setChillerFlowMode('ConstantFlow')
            chiller.setSizingFactor(chiller_sizing_factor)
            chiller.setReferenceCOP(round(default_cop, 3))

            # connect the chiller to the condenser loop if one was supplied
            if condenser_water_loop is None:
                chiller.setCondenserType('AirCooled')
            else:
                condenser_water_loop.addDemandBranchForComponent(chiller)
                chiller.setCondenserType('WaterCooled')

    # enable waterside economizer if requested
    if condenser_water_loop is not None:
        if waterside_economizer == 'integrated':
            model_add_waterside_economizer(
                model, chilled_water_loop, condenser_water_loop, integrated=True)
        elif waterside_economizer == 'non-integrated':
            model_add_waterside_economizer(
                model, chilled_water_loop, condenser_water_loop, integrated=False)

    # chilled water loop pipes
    chiller_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    chiller_bypass_pipe.setName('{} Chiller Bypass'.format(chilled_water_loop.nameString()))
    chilled_water_loop.addSupplyBranchForComponent(chiller_bypass_pipe)

    coil_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    coil_bypass_pipe.setName('{} Coil Bypass'.format(chilled_water_loop.nameString()))
    chilled_water_loop.addDemandBranchForComponent(coil_bypass_pipe)

    supply_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    supply_outlet_pipe.setName('{} Supply Outlet'.format(chilled_water_loop.nameString()))
    supply_outlet_pipe.addToNode(chilled_water_loop.supplyOutletNode())

    demand_inlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_inlet_pipe.setName('{} Demand Inlet'.format(chilled_water_loop.nameString()))
    demand_inlet_pipe.addToNode(chilled_water_loop.demandInletNode())

    demand_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_outlet_pipe.setName('{} Demand Outlet'.format(chilled_water_loop.nameString()))
    demand_outlet_pipe.addToNode(chilled_water_loop.demandOutletNode())

    return chilled_water_loop


def model_add_vsd_twr_fan_curve(model):
    """Add a curve to be used for cooling tower fans."""
    # check for the existing curve
    exist_curve = model.getCurveCubicByName('VSD-TWR-FAN-FPLR')
    if exist_curve.is_initialized():
        return exist_curve.get()
    # create the curve
    curve = openstudio_model.CurveCubic(model)
    curve.setName('VSD-TWR-FAN-FPLR')
    curve.setCoefficient1Constant(0.33162901)
    curve.setCoefficient2x(-0.88567609)
    curve.setCoefficient3xPOW2(0.60556507)
    curve.setCoefficient4xPOW3(0.9484823)
    curve.setMinimumValueofx(0.0)
    curve.setMaximumValueofx(1.0)
    return curve


def model_add_cw_loop(
        model, system_name='Condenser Water Loop', cooling_tower_type='Open Cooling Tower',
        cooling_tower_fan_type='Propeller or Axial',
        cooling_tower_capacity_control='TwoSpeed Fan',
        number_of_cells_per_tower=1, number_cooling_towers=1, use_90_1_design_sizing=True,
        sup_wtr_temp=70.0, dsgn_sup_wtr_temp=85.0, dsgn_sup_wtr_temp_delt=10.0,
        wet_bulb_approach=7.0, pump_spd_ctrl='Constant', pump_tot_hd=49.7):
    """Creates a condenser water loop and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        cooling_tower_type: [String] valid choices are Open Cooling Tower,
            Closed Cooling Tower.
        cooling_tower_fan_type: [String] valid choices are Centrifugal, "Propeller or Axial."
        cooling_tower_capacity_control: [String] valid choices are Fluid Bypass,
            Fan Cycling, TwoSpeed Fan, Variable Speed Fan.
        number_of_cells_per_tower: [Integer] the number of discrete cells per tower.
        number_cooling_towers: [Integer] the number of cooling towers to be
            added (in parallel).
        use_90_1_design_sizing: [Boolean] will determine the design sizing
            temperatures based on the 90.1 Appendix G approach. Overrides sup_wtr_temp,
            dsgn_sup_wtr_temp, dsgn_sup_wtr_temp_delt, and wet_bulb_approach if True.
        sup_wtr_temp: [Double] supply water temperature in degrees
            Fahrenheit, default 70F.
        dsgn_sup_wtr_temp: [Double] design supply water temperature in degrees
            Fahrenheit, default 85F.
        dsgn_sup_wtr_temp_delt: [Double] design water range temperature in
            degrees Rankine (aka. dF), default 10R.
        wet_bulb_approach: [Double] design wet bulb approach temperature, default 7R.
        pump_spd_ctrl: [String] pump speed control type, Constant or Variable (default).
        pump_tot_hd: [Double] pump head in ft H2O.

    Returns:
        [OpenStudio::Model::PlantLoop] the resulting condenser water plant loop
    """
    # create condenser water loop
    condenser_water_loop = openstudio_model.PlantLoop(model)
    system_name = 'Condenser Water Loop' if system_name is None else system_name
    condenser_water_loop.setName(system_name)

    # condenser water loop sizing and controls
    sup_wtr_temp = 70.0 if sup_wtr_temp is None else sup_wtr_temp
    sup_wtr_temp_c = TEMPERATURE.to_unit([sup_wtr_temp], 'C', 'F')[0]
    dsgn_sup_wtr_temp = 85.0 if dsgn_sup_wtr_temp is None else dsgn_sup_wtr_temp
    dsgn_sup_wtr_temp_c = TEMPERATURE.to_unit([dsgn_sup_wtr_temp], 'C', 'F')[0]
    dsgn_sup_wtr_temp_delt = 10.0 if dsgn_sup_wtr_temp_delt is None \
        else dsgn_sup_wtr_temp_delt
    dsgn_sup_wtr_temp_delt_k = TEMP_DELTA.to_unit([dsgn_sup_wtr_temp_delt], 'dC', 'dF')[0]
    wet_bulb_approach = 7.0 if wet_bulb_approach is None else wet_bulb_approach
    wet_bulb_approach_k = TEMP_DELTA.to_unit([wet_bulb_approach], 'dC', 'dF')[0]

    condenser_water_loop.setMinimumLoopTemperature(5.0)
    condenser_water_loop.setMaximumLoopTemperature(80.0)
    sizing_plant = condenser_water_loop.sizingPlant()
    sizing_plant.setLoopType('Condenser')
    sizing_plant.setDesignLoopExitTemperature(dsgn_sup_wtr_temp_c)
    sizing_plant.setLoopDesignTemperatureDifference(dsgn_sup_wtr_temp_delt_k)
    sizing_plant.setSizingOption('Coincident')
    sizing_plant.setZoneTimestepsinAveragingWindow(6)
    sizing_plant.setCoincidentSizingFactorMode('GlobalCoolingSizingFactor')

    # follow outdoor air wetbulb with given approach temperature
    cw_stpt_manager = openstudio_model.SetpointManagerFollowOutdoorAirTemperature(model)
    s_pt_name = '{} Setpoint Manager Follow OATwb with {}F Approach'.format(
        condenser_water_loop.nameString(), wet_bulb_approach)
    cw_stpt_manager.setName(s_pt_name)
    cw_stpt_manager.setReferenceTemperatureType('OutdoorAirWetBulb')
    cw_stpt_manager.setMaximumSetpointTemperature(dsgn_sup_wtr_temp_c)
    cw_stpt_manager.setMinimumSetpointTemperature(sup_wtr_temp_c)
    cw_stpt_manager.setOffsetTemperatureDifference(wet_bulb_approach_k)
    cw_stpt_manager.addToNode(condenser_water_loop.supplyOutletNode())

    # create condenser water pump
    if pump_spd_ctrl == 'Constant':
        cw_pump = openstudio_model.PumpConstantSpeed(model)
    elif pump_spd_ctrl == 'Variable':
        cw_pump = openstudio_model.PumpVariableSpeed(model)
    elif pump_spd_ctrl == 'HeaderedVariable':
        cw_pump = openstudio_model.HeaderedPumpsVariableSpeed(model)
        cw_pump.setNumberofPumpsinBank(2)
    elif pump_spd_ctrl == 'HeaderedConstant':
        cw_pump = openstudio_model.HeaderedPumpsConstantSpeed(model)
        cw_pump.setNumberofPumpsinBank(2)
    else:
        cw_pump = openstudio_model.PumpConstantSpeed(model)
    cw_pump.setName('{} {} Pump'.format(condenser_water_loop.nameString(), pump_spd_ctrl))
    cw_pump.setPumpControlType('Intermittent')

    pump_tot_hd = 49.7 if pump_tot_hd is None else pump_tot_hd
    pump_tot_hd_pa = PRESSURE.to_unit([pump_tot_hd * 12], 'Pa', 'inH2O')[0]
    cw_pump.setRatedPumpHead(pump_tot_hd_pa)
    cw_pump.addToNode(condenser_water_loop.supplyInletNode())

    # Cooling towers
    # Per PNNL PRM Reference Manual
    for _ in range(number_cooling_towers):
        # Tower object depends on the control type
        cooling_tower = None
        if cooling_tower_capacity_control in ('Fluid Bypass', 'Fan Cycling'):
            cooling_tower = openstudio_model.CoolingTowerSingleSpeed(model)
            if cooling_tower_capacity_control == 'Fluid Bypass':
                cooling_tower.setCellControl('FluidBypass')
            else:
                cooling_tower.setCellControl('FanCycling')
        elif cooling_tower_capacity_control == 'TwoSpeed Fan':
            cooling_tower = openstudio_model.CoolingTowerTwoSpeed(model)
        elif cooling_tower_capacity_control == 'Variable Speed Fan':
            cooling_tower = openstudio_model.CoolingTowerVariableSpeed(model)
            cooling_tower.setDesignRangeTemperature(dsgn_sup_wtr_temp_delt_k)
            cooling_tower.setDesignApproachTemperature(wet_bulb_approach_k)
            cooling_tower.setFractionofTowerCapacityinFreeConvectionRegime(0.125)
            twr_fan_curve = model_add_vsd_twr_fan_curve(model)
            cooling_tower.setFanPowerRatioFunctionofAirFlowRateRatioCurve(twr_fan_curve)
        else:
            msg = '{} is not a valid choice of cooling tower capacity control. ' \
                'Valid choices are Fluid Bypass, Fan Cycling, TwoSpeed Fan, Variable ' \
                'Speed Fan.'.format(cooling_tower_capacity_control)
            print(msg)

        # Set the properties that apply to all tower types and attach to the condenser loop.
        if cooling_tower is not None:
            twr_name = '{} {} {}'.format(
                cooling_tower_fan_type, cooling_tower_capacity_control,
                cooling_tower_type)
            cooling_tower.setName(twr_name)
            cooling_tower.setSizingFactor(1 / number_cooling_towers)
            cooling_tower.setNumberofCells(number_of_cells_per_tower)
            condenser_water_loop.addSupplyBranchForComponent(cooling_tower)

    # apply 90.1 sizing temperatures
    if use_90_1_design_sizing:
        # use the formulation in 90.1-2010 G3.1.3.11 to set the approach temperature
        # first, look in the model design day objects for sizing information
        summer_oat_wbs_f = []
        for dd in model.getDesignDays():
            if dd.dayType != 'SummerDesignDay':
                continue
            if 'WB=>MDB' not in dd.nameString():
                continue

            if model.version() < openstudio.VersionString('3.3.0'):
                if dd.humidityIndicatingType == 'Wetbulb':
                    summer_oat_wb_c = dd.humidityIndicatingConditionsAtMaximumDryBulb()
                    summer_oat_wbs_f.append(TEMPERATURE.to_unit([summer_oat_wb_c], 'F', 'C')[0])
                else:
                    msg = 'For {}, humidity is specified as {}; cannot determine Twb.'.format(
                        dd.nameString, dd.humidityIndicatingType())
                    print(msg)
            else:
                if dd.humidityConditionType() == 'Wetbulb' and \
                        dd.wetBulbOrDewPointAtMaximumDryBulb().is_initialized():
                    wb_mdbt = dd.wetBulbOrDewPointAtMaximumDryBulb().get()
                    summer_oat_wbs_f.append(TEMPERATURE.to_unit([wb_mdbt], 'F', 'C')[0])
                else:
                    msg = 'For {}, humidity is specified as {}; cannot determine Twb.'.format(
                        dd.nameString(), dd.humidityConditionType())
                    print(msg)

        # if values are still absent, use the CTI rating condition 78F
        design_oat_wb_f = None
        if len(summer_oat_wbs_f) == 0:
            design_oat_wb_f = 78.0
        else:
            design_oat_wb_f = max(summer_oat_wbs_f)  # Take worst case condition
        design_oat_wb_c = TEMPERATURE.to_unit([design_oat_wb_f], 'C', 'F')[0]

        # call method to apply design sizing to the condenser water loop
        prototype_apply_condenser_water_temperatures(
            condenser_water_loop, design_wet_bulb_c=design_oat_wb_c)

    # Condenser water loop pipes
    cooling_tower_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    pipe_name = '{} Cooling Tower Bypass'.format(condenser_water_loop.nameString())
    cooling_tower_bypass_pipe.setName(pipe_name)
    condenser_water_loop.addSupplyBranchForComponent(cooling_tower_bypass_pipe)

    chiller_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    pipe_name = '{} Chiller Bypass'.format(condenser_water_loop.nameString())
    chiller_bypass_pipe.setName(pipe_name)
    condenser_water_loop.addDemandBranchForComponent(chiller_bypass_pipe)

    supply_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    supply_outlet_pipe.setName('{} Supply Outlet'.format(condenser_water_loop.nameString()))
    supply_outlet_pipe.addToNode(condenser_water_loop.supplyOutletNode())

    demand_inlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_inlet_pipe.setName('{} Demand Inlet'.format(condenser_water_loop.nameString()))
    demand_inlet_pipe.addToNode(condenser_water_loop.demandInletNode())

    demand_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_outlet_pipe.setName('{} Demand Outlet'.format(condenser_water_loop.nameString()))
    demand_outlet_pipe.addToNode(condenser_water_loop.demandOutletNode())

    return condenser_water_loop


def model_add_hp_loop(
        model, heating_fuel='NaturalGas', cooling_fuel='Electricity',
        cooling_type='EvaporativeFluidCooler', system_name='Heat Pump Loop',
        sup_wtr_high_temp=87.0, sup_wtr_low_temp=67.0,
        dsgn_sup_wtr_temp=102.2, dsgn_sup_wtr_temp_delt=19.8):
    """Creates a heat pump loop which has a boiler and fluid cooler.

    Args:
        model [OpenStudio::Model::Model] OpenStudio model object.
        heating_fuel: [String]
        cooling_fuel: [String] cooling fuel. Valid options are: Electricity,
            DistrictCooling.
        cooling_type: [String] cooling type if not DistrictCooling.
            Valid options are: CoolingTower, CoolingTowerSingleSpeed,
            CoolingTowerTwoSpeed, CoolingTowerVariableSpeed, FluidCooler,
            FluidCoolerSingleSpeed, FluidCoolerTwoSpeed, EvaporativeFluidCooler,
            EvaporativeFluidCoolerSingleSpeed, EvaporativeFluidCoolerTwoSpeed
        system_name: [String] the name of the system, or None in which case it
            will be defaulted
        sup_wtr_high_temp: [Double] target supply water temperature to enable
            cooling in degrees Fahrenheit, default 65.0F
        sup_wtr_low_temp: [Double] target supply water temperature to enable
            heating in degrees Fahrenheit, default 41.0F
        dsgn_sup_wtr_temp: [Double] design supply water temperature in degrees
            Fahrenheit, default 102.2F
        dsgn_sup_wtr_temp_delt: [Double] design supply-return water temperature
            difference in degrees Rankine, default 19.8R.

    Returns:
        [OpenStudio::Model::PlantLoop] the resulting plant loop.
    """
    # create heat pump loop
    heat_pump_water_loop = openstudio_model.PlantLoop(model)
    heat_pump_water_loop.setLoadDistributionScheme('SequentialLoad')
    if system_name is None:
        heat_pump_water_loop.setName('Heat Pump Loop')
    else:
        heat_pump_water_loop.setName(system_name)

    # hot water loop sizing and controls
    sup_wtr_high_temp = 87.0 if sup_wtr_high_temp is None else sup_wtr_high_temp
    sup_wtr_high_temp_c = TEMPERATURE.to_unit([sup_wtr_high_temp], 'C', 'F')[0]
    sup_wtr_low_temp = 67.0 if sup_wtr_low_temp is None else sup_wtr_low_temp
    sup_wtr_low_temp_c = TEMPERATURE.to_unit([sup_wtr_low_temp], 'C', 'F')[0]
    dsgn_sup_wtr_temp = 102.2 if dsgn_sup_wtr_temp is None else dsgn_sup_wtr_temp
    dsgn_sup_wtr_temp_c = TEMPERATURE.to_unit([dsgn_sup_wtr_temp], 'C', 'F')[0]
    dsgn_sup_wtr_temp_delt = 19.8 if dsgn_sup_wtr_temp_delt is None \
        else dsgn_sup_wtr_temp_delt
    dsgn_sup_wtr_temp_delt_k = TEMP_DELTA.to_unit([dsgn_sup_wtr_temp_delt], 'dC', 'dF')[0]

    sizing_plant = heat_pump_water_loop.sizingPlant()
    sizing_plant.setLoopType('Heating')
    heat_pump_water_loop.setMinimumLoopTemperature(10.0)
    heat_pump_water_loop.setMaximumLoopTemperature(35.0)
    sizing_plant.setDesignLoopExitTemperature(dsgn_sup_wtr_temp_c)
    sizing_plant.setLoopDesignTemperatureDifference(dsgn_sup_wtr_temp_delt_k)
    loop_name = heat_pump_water_loop.nameString()
    hp_high_temp_sch = create_constant_schedule_ruleset(
        model, sup_wtr_high_temp_c,
        name='{} High Temp - {}F'.format(loop_name, int(sup_wtr_high_temp)),
        schedule_type_limit='Temperature')
    hp_low_temp_sch = create_constant_schedule_ruleset(
        model, sup_wtr_low_temp_c,
        name='{} Low Temp - {}F'.format(loop_name, int(sup_wtr_low_temp)),
        schedule_type_limit='Temperature')
    hp_stpt_manager = openstudio_model.SetpointManagerScheduledDualSetpoint(model)
    hp_stpt_manager.setName('{} Scheduled Dual Setpoint'.format(loop_name))
    hp_stpt_manager.setHighSetpointSchedule(hp_high_temp_sch)
    hp_stpt_manager.setLowSetpointSchedule(hp_low_temp_sch)
    hp_stpt_manager.addToNode(heat_pump_water_loop.supplyOutletNode())

    # create pump
    hp_pump = openstudio_model.PumpConstantSpeed(model)
    hp_pump.setName('{} Pump'.format(loop_name))
    hp_pump.setRatedPumpHead(PRESSURE.to_unit([60.0 * 12], 'Pa', 'inH2O')[0])
    hp_pump.setPumpControlType('Intermittent')
    hp_pump.addToNode(heat_pump_water_loop.supplyInletNode())

    # add setpoint to cooling outlet so correct plant operation scheme is generated
    cooling_equipment_stpt_manager = \
        openstudio_model.SetpointManagerScheduledDualSetpoint(model)
    cooling_equipment_stpt_manager.setHighSetpointSchedule(hp_high_temp_sch)
    cooling_equipment_stpt_manager.setLowSetpointSchedule(hp_low_temp_sch)

    # create cooling equipment and add to the loop
    if cooling_fuel == 'DistrictCooling':
        cooling_equipment = openstudio_model.DistrictCooling(model)
        cooling_equipment.setName('{} District Cooling'.format(loop_name))
        cooling_equipment.autosizeNominalCapacity()
        heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
        cooling_equipment_stpt_manager.setName(
            '{} District Cooling Scheduled Dual Setpoint'.format(loop_name))
    else:
        if cooling_type in ('CoolingTower', 'CoolingTowerTwoSpeed'):
            cooling_equipment = openstudio_model.CoolingTowerTwoSpeed(model)
            cooling_equipment.setName('{} CoolingTowerTwoSpeed'.format(loop_name))
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Cooling Tower Scheduled Dual Setpoint'.format(loop_name))
        elif cooling_type == 'CoolingTowerSingleSpeed':
            cooling_equipment = openstudio_model.CoolingTowerSingleSpeed(model)
            cooling_equipment.setName('{} CoolingTowerSingleSpeed'.format(loop_name))
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Cooling Tower Scheduled Dual Setpoint'.format(loop_name))
        elif cooling_type == 'CoolingTowerVariableSpeed':
            cooling_equipment = openstudio_model.CoolingTowerVariableSpeed(model)
            cooling_equipment.setName('{} CoolingTowerVariableSpeed'.format(loop_name))
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Cooling Tower Scheduled Dual Setpoint'.format(loop_name))
        elif cooling_type in ('FluidCooler', 'FluidCoolerSingleSpeed'):
            cooling_equipment = openstudio_model.FluidCoolerSingleSpeed(model)
            cooling_equipment.setName('{} FluidCoolerSingleSpeed'.format(loop_name))
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Fluid Cooler Scheduled Dual Setpoint'.format(loop_name))
            # Remove hard coded default values
            cooling_equipment.setPerformanceInputMethod(
                'UFactorTimesAreaAndDesignWaterFlowRate')
            cooling_equipment.autosizeDesignWaterFlowRate()
            cooling_equipment.autosizeDesignAirFlowRate()
        elif cooling_type == 'FluidCoolerTwoSpeed':
            cooling_equipment = openstudio_model.FluidCoolerTwoSpeed(model)
            cooling_equipment.setName('{} FluidCoolerTwoSpeed'.format(loop_name))
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Fluid Cooler Scheduled Dual Setpoint'.format(loop_name))
            # Remove hard coded default values
            cooling_equipment.setPerformanceInputMethod(
                'UFactorTimesAreaAndDesignWaterFlowRate')
            cooling_equipment.autosizeDesignWaterFlowRate()
            cooling_equipment.autosizeHighFanSpeedAirFlowRate()
            cooling_equipment.autosizeLowFanSpeedAirFlowRate()
        elif cooling_type in ('EvaporativeFluidCooler', 'EvaporativeFluidCoolerSingleSpeed'):
            cooling_equipment = openstudio_model.EvaporativeFluidCoolerSingleSpeed(model)
            cooling_equipment.setName(
                '{} EvaporativeFluidCoolerSingleSpeed'.format(loop_name))
            cooling_equipment.setDesignSprayWaterFlowRate(0.002208)  # Based on HighRiseApartment
            cooling_equipment.setPerformanceInputMethod(
                'UFactorTimesAreaAndDesignWaterFlowRate')
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Fluid Cooler Scheduled Dual Setpoint'.format(loop_name))
        elif cooling_type == 'EvaporativeFluidCoolerTwoSpeed':
            cooling_equipment = openstudio_model.EvaporativeFluidCoolerTwoSpeed(model)
            cooling_equipment.setName('{} EvaporativeFluidCoolerTwoSpeed'.format(loop_name))
            cooling_equipment.setDesignSprayWaterFlowRate(0.002208)  # Based on HighRiseApartment
            cooling_equipment.setPerformanceInputMethod(
                'UFactorTimesAreaAndDesignWaterFlowRate')
            heat_pump_water_loop.addSupplyBranchForComponent(cooling_equipment)
            cooling_equipment_stpt_manager.setName(
                '{} Fluid Cooler Scheduled Dual Setpoint'.format(loop_name))
        else:
            msg = 'Cooling fuel type "{}" is not a valid option.'.format(cooling_type)
            raise ValueError(msg)
    equip_out_node = cooling_equipment.outletModelObject().get().to_Node().get()
    cooling_equipment_stpt_manager.addToNode(equip_out_node)

    # add setpoint to heating outlet so correct plant operation scheme is generated
    heating_equipment_stpt_manager = \
        openstudio_model.SetpointManagerScheduledDualSetpoint(model)
    heating_equipment_stpt_manager.setHighSetpointSchedule(hp_high_temp_sch)
    heating_equipment_stpt_manager.setLowSetpointSchedule(hp_low_temp_sch)

    # switch statement to handle district heating name change
    if model.version() < openstudio.VersionString('3.7.0'):
        if heating_fuel == 'DistrictHeatingWater' or \
                heating_fuel == 'DistrictHeatingSteam':
            heating_fuel = 'DistrictHeating'
    else:
        if heating_fuel == 'DistrictHeating':
            heating_fuel = 'DistrictHeatingWater'

    # create heating equipment and add to the loop
    if heating_fuel == 'DistrictHeating':
        heating_equipment = openstudio_model.DistrictHeating(model)
        heating_equipment.setName('{} District Heating'.format(loop_name))
        heating_equipment.autosizeNominalCapacity()
        heat_pump_water_loop.addSupplyBranchForComponent(heating_equipment)
        heating_equipment_stpt_manager.setName(
            '{} District Heating Scheduled Dual Setpoint'.format(loop_name))
    elif heating_fuel == 'DistrictHeatingWater':
        heating_equipment = openstudio_model.DistrictHeatingWater(model)
        heating_equipment.setName('{} District Heating'.format(loop_name))
        heating_equipment.autosizeNominalCapacity()
        heat_pump_water_loop.addSupplyBranchForComponent(heating_equipment)
        heating_equipment_stpt_manager.setName(
            '{} District Heating Scheduled Dual Setpoint'.format(loop_name))
    elif heating_fuel == 'DistrictHeatingSteam':
        heating_equipment = openstudio_model.DistrictHeatingSteam(model)
        heating_equipment.setName('{} District Heating'.format(loop_name))
        heating_equipment.autosizeNominalCapacity()
        heat_pump_water_loop.addSupplyBranchForComponent(heating_equipment)
        heating_equipment_stpt_manager.setName(
            '{} District Heating Scheduled Dual Setpoint'.format(loop_name))
    elif heating_fuel in ('AirSourceHeatPump', 'ASHP'):
        heating_equipment = create_central_air_source_heat_pump(
            model, heat_pump_water_loop)
        heating_equipment_stpt_manager.setName(
            '{} ASHP Scheduled Dual Setpoint'.format(loop_name))
    elif heating_fuel in ('Electricity', 'Gas', 'NaturalGas', 'Propane',
                          'PropaneGas', 'FuelOilNo1', 'FuelOilNo2'):
        heating_equipment = create_boiler_hot_water(
            model, hot_water_loop=heat_pump_water_loop,
            name='{} Supplemental Boiler'.format(loop_name), fuel_type=heating_fuel,
            flow_mode='ConstantFlow',
            lvg_temp_dsgn_f=86.0, min_plr=0.0, max_plr=1.2, opt_plr=1.0)
        heating_equipment_stpt_manager.setName(
            '{} Boiler Scheduled Dual Setpoint'.format(loop_name))
    else:
        raise ValueError('Boiler fuel type "{}" is not valid'.format(heating_fuel))
    equip_out_node = heating_equipment.outletModelObject().get().to_Node().get()
    heating_equipment_stpt_manager.addToNode(equip_out_node)

    # add heat pump water loop pipes
    supply_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    supply_bypass_pipe.setName('{} Supply Bypass'.format(loop_name))
    heat_pump_water_loop.addSupplyBranchForComponent(supply_bypass_pipe)

    demand_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    demand_bypass_pipe.setName('{} Demand Bypass'.format(loop_name))
    heat_pump_water_loop.addDemandBranchForComponent(demand_bypass_pipe)

    supply_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    supply_outlet_pipe.setName('{} Supply Outlet'.format(loop_name))
    supply_outlet_pipe.addToNode(heat_pump_water_loop.supplyOutletNode())

    demand_inlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_inlet_pipe.setName('{} Demand Inlet'.format(loop_name))
    demand_inlet_pipe.addToNode(heat_pump_water_loop.demandInletNode())

    demand_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_outlet_pipe.setName('{} Demand Outlet'.format(loop_name))
    demand_outlet_pipe.addToNode(heat_pump_water_loop.demandOutletNode())

    return heat_pump_water_loop


def model_add_ground_hx_loop(model, system_name='Ground HX Loop'):
    """Creates loop that roughly mimics a properly sized ground heat exchanger.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object
        system_name: [String] the name of the system, or None in which case
            it will be defaulted.

    Returns:
        [OpenStudio::Model::PlantLoop] the resulting plant loop.
    """
    # create ground hx loop
    ground_hx_loop = openstudio_model.PlantLoop(model)
    system_name = 'Ground HX Loop' if system_name is None else system_name
    ground_hx_loop.setName(system_name)
    loop_name = ground_hx_loop.nameString()

    # ground hx loop sizing and controls
    ground_hx_loop.setMinimumLoopTemperature(5.0)
    ground_hx_loop.setMaximumLoopTemperature(80.0)
    # temp change at high and low entering condition
    delta_t_k = TEMP_DELTA.to_unit([12.0], 'dC', 'dF')[0]
    # low entering condition
    min_inlet_c = TEMPERATURE.to_unit([30.0], 'C', 'F')[0]
    # high entering condition
    max_inlet_c = TEMPERATURE.to_unit([90.0], 'C', 'F')[0]

    # calculate the linear formula that defines outlet temperature
    # based on inlet temperature of the ground hx
    min_outlet_c = min_inlet_c + delta_t_k
    max_outlet_c = max_inlet_c - delta_t_k
    slope_c_per_c = (max_outlet_c - min_outlet_c) / (max_inlet_c - min_inlet_c)
    intercept_c = min_outlet_c - (slope_c_per_c * min_inlet_c)

    sizing_plant = ground_hx_loop.sizingPlant()
    sizing_plant.setLoopType('Heating')
    sizing_plant.setDesignLoopExitTemperature(max_outlet_c)
    sizing_plant.setLoopDesignTemperatureDifference(delta_t_k)

    # create pump
    pump = openstudio_model.PumpConstantSpeed(model)
    pump.setName('{} Pump'.format(loop_name))
    pump.setRatedPumpHead(PRESSURE.to_unit([60.0 * 12], 'Pa', 'inH2O')[0])
    pump.setPumpControlType('Intermittent')
    pump.addToNode(ground_hx_loop.supplyInletNode())

    # use EMS and a PlantComponentTemperatureSource
    # to mimic the operation of the ground heat exchanger

    # schedule to actuate ground HX outlet temperature
    hx_temp_sch = openstudio_model.ScheduleConstant(model)
    hx_temp_sch.setName('Ground HX Temp Sch')
    hx_temp_sch.setValue(24.0)

    ground_hx = openstudio_model.PlantComponentTemperatureSource(model)
    ground_hx.setName('Ground HX')
    ground_hx.setTemperatureSpecificationType('Scheduled')
    ground_hx.setSourceTemperatureSchedule(hx_temp_sch)
    ground_hx_loop.addSupplyBranchForComponent(ground_hx)

    hx_stpt_manager = openstudio_model.SetpointManagerScheduled(model, hx_temp_sch)
    hx_stpt_manager.setName('{} Supply Outlet Setpoint'.format(ground_hx.nameString()))
    hx_stpt_manager.addToNode(ground_hx.outletModelObject().get().to_Node().get())

    loop_stpt_manager = openstudio_model.SetpointManagerScheduled(model, hx_temp_sch)
    loop_stpt_manager.setName('{} Supply Outlet Setpoint'.format(ground_hx_loop.nameString()))
    loop_stpt_manager.addToNode(ground_hx_loop.supplyOutletNode())

    # edit name to be EMS friendly
    ground_hx_ems_name = ems_friendly_name(ground_hx.nameString())

    # sensor to read supply inlet temperature
    inlet_temp_sensor = openstudio_model.EnergyManagementSystemSensor(
        model, 'System Node Temperature')
    inlet_temp_sensor.setName('{} Inlet Temp Sensor'.format(ground_hx_ems_name))
    inlet_temp_sensor.setKeyName(str(ground_hx_loop.supplyInletNode().handle()))

    # actuator to set supply outlet temperature
    outlet_temp_actuator = openstudio_model.EnergyManagementSystemActuator(
        hx_temp_sch, 'Schedule:Constant', 'Schedule Value')
    outlet_temp_actuator.setName('{} Outlet Temp Actuator'.format(ground_hx_ems_name))

    # program to control outlet temperature
    # adjusts delta-t based on calculation of slope and intercept from control temperatures
    program = openstudio_model.EnergyManagementSystemProgram(model)
    program.setName('{} Temperature Control'.format(ground_hx_ems_name))
    program_body = \
        'SET Tin = {inlet_temp_sensor_handle}\n' \
        'SET Tout = {slope_c_per_c} * Tin + {intercept_c}\n' \
        'SET {outlet_temp_actuator_handle} = Tout'.format(
            inlet_temp_sensor_handle=inlet_temp_sensor.handle(),
            slope_c_per_c=round(slope_c_per_c, 2), intercept_c=round(intercept_c, 2),
            outlet_temp_actuator_handle=outlet_temp_actuator.handle()
        )
    program.setBody(program_body)

    # program calling manager
    pcm = openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    pcm.setName('{} Calling Manager'.format(program.nameString()))
    pcm.setCallingPoint('InsideHVACSystemIterationLoop')
    pcm.addProgram(program)

    return ground_hx_loop


def model_add_district_ambient_loop(model, system_name='Ambient Loop'):
    """Adds an ambient condenser water loop that represents a district system.

    It connects buildings as a shared sink/source for heat pumps.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.

    Returns:
        [OpenStudio::Model::PlantLoop] the ambient loop.
    """
    # create ambient loop
    ambient_loop = openstudio_model.PlantLoop(model)
    system_name = 'Ambient Loop' if system_name is None else system_name
    ambient_loop.setName(system_name)
    loop_name = ambient_loop.nameString()

    # ambient loop sizing and controls
    ambient_loop.setMinimumLoopTemperature(5.0)
    ambient_loop.setMaximumLoopTemperature(80.0)

    amb_high_temp_f = 90  # Supplemental cooling below 65F
    amb_low_temp_f = 41  # Supplemental heat below 41F
    amb_temp_sizing_f = 102.2  # CW sized to deliver 102.2F
    amb_delta_t_r = 19.8  # 19.8F delta-T
    amb_high_temp_c = TEMPERATURE.to_unit([amb_high_temp_f], 'C', 'F')[0]
    amb_low_temp_c = TEMPERATURE.to_unit([amb_low_temp_f], 'C', 'F')[0]
    amb_temp_sizing_c = TEMPERATURE.to_unit([amb_temp_sizing_f], 'C', 'F')[0]
    amb_delta_t_k = TEMP_DELTA.to_unit([amb_delta_t_r], 'dC', 'dF')[0]

    amb_high_temp_sch = create_constant_schedule_ruleset(
        model, amb_high_temp_c,
        name='Ambient Loop High Temp - {}F'.format(amb_high_temp_f),
        schedule_type_limit='Temperature')
    amb_low_temp_sch = create_constant_schedule_ruleset(
        model, amb_low_temp_c,
        name='Ambient Loop Low Temp - {}F'.format(amb_low_temp_f),
        schedule_type_limit='Temperature')

    amb_stpt_manager = openstudio_model.SetpointManagerScheduledDualSetpoint(model)
    amb_stpt_manager.setName('{} Supply Water Setpoint Manager'.format(loop_name))
    amb_stpt_manager.setHighSetpointSchedule(amb_high_temp_sch)
    amb_stpt_manager.setLowSetpointSchedule(amb_low_temp_sch)
    amb_stpt_manager.addToNode(ambient_loop.supplyOutletNode())

    sizing_plant = ambient_loop.sizingPlant()
    sizing_plant.setLoopType('Heating')
    sizing_plant.setDesignLoopExitTemperature(amb_temp_sizing_c)
    sizing_plant.setLoopDesignTemperatureDifference(amb_delta_t_k)

    # create pump
    pump = openstudio_model.PumpVariableSpeed(model)
    pump.setName('{} Pump'.format(loop_name))
    pump.setRatedPumpHead(PRESSURE.to_unit([60.0 * 12], 'Pa', 'inH2O')[0])
    pump.setPumpControlType('Intermittent')
    pump.addToNode(ambient_loop.supplyInletNode())

    # cooling
    district_cooling = openstudio_model.DistrictCooling(model)
    district_cooling.setNominalCapacity(1000000000000)  # large number; no autosizing
    ambient_loop.addSupplyBranchForComponent(district_cooling)

    # heating
    if model.version() < openstudio.VersionString('3.7.0'):
        district_heating = openstudio_model.DistrictHeating(model)
    else:
        district_heating = openstudio_model.DistrictHeatingWater(model)
    district_heating.setNominalCapacity(1000000000000)  # large number; no autosizing
    ambient_loop.addSupplyBranchForComponent(district_heating)

    # add ambient water loop pipes
    supply_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    supply_bypass_pipe.setName('{} Supply Bypass'.format(loop_name))
    ambient_loop.addSupplyBranchForComponent(supply_bypass_pipe)

    demand_bypass_pipe = openstudio_model.PipeAdiabatic(model)
    demand_bypass_pipe.setName('{} Demand Bypass'.format(loop_name))
    ambient_loop.addDemandBranchForComponent(demand_bypass_pipe)

    supply_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    supply_outlet_pipe.setName('{} Supply Outlet'.format(loop_name))
    supply_outlet_pipe.addToNode(ambient_loop.supplyOutletNode)

    demand_inlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_inlet_pipe.setName('{} Demand Inlet'.format(loop_name))
    demand_inlet_pipe.addToNode(ambient_loop.demandInletNode)

    demand_outlet_pipe = openstudio_model.PipeAdiabatic(model)
    demand_outlet_pipe.setName('{} Demand Outlet'.format(loop_name))
    demand_outlet_pipe.addToNode(ambient_loop.demandOutletNode)

    return ambient_loop


def model_add_doas_cold_supply(
        model, thermal_zones, system_name=None, hot_water_loop=None,
        chilled_water_loop=None, hvac_op_sch=None, min_oa_sch=None, min_frac_oa_sch=None,
        fan_maximum_flow_rate=None, econo_ctrl_mthd='FixedDryBulb',
        energy_recovery=False, clg_dsgn_sup_air_temp=55.0, htg_dsgn_sup_air_temp=60.0):
    """Creates a DOAS system with cold supply and terminal units for each zone.

    This is the default DOAS system for DOE prototype buildings.
    Use model_add_doas for other DOAS systems.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to connect
            to heating and zone fan coils.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop to
            connect to cooling coil.
        hvac_op_sch: [String] name of the HVAC operation schedule, default is always on.
        min_oa_sch: [String] name of the minimum outdoor air schedule, default
            is always on.
        min_frac_oa_sch: [String] name of the minimum fraction of outdoor air
            schedule, default is always on.
        fan_maximum_flow_rate: [Double] fan maximum flow rate in cfm, default
            is autosize.
        econo_ctrl_mthd: [String] economizer control type, default is Fixed Dry Bulb.
        energy_recovery: [Boolean] if true, an ERV will be added to the system.
        clg_dsgn_sup_air_temp: [Double] design cooling supply air temperature
            in degrees Fahrenheit, default 65F.
        htg_dsgn_sup_air_temp: [Double] design heating supply air temperature
            in degrees Fahrenheit, default 75F.

    Returns:
        [OpenStudio::Model::AirLoopHVAC] the resulting DOAS air loop.
    """
    # Check the total OA requirement for all zones on the system
    tot_oa_req = 0
    for zone in thermal_zones:
        tot_oa_req += thermal_zone_get_outdoor_airflow_rate(zone)
        if tot_oa_req > 0:
            break

    # If the total OA requirement is zero do not add the DOAS system
    # because the simulations will fail
    if tot_oa_req == 0:
        return None

    # create a DOAS air loop
    air_loop = openstudio_model.AirLoopHVAC(model)
    system_name = '{} Zone DOAS'.format(len(thermal_zones)) \
        if system_name is None else system_name
    air_loop.setName(system_name)
    loop_name = air_loop.nameString()

    # set availability schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # DOAS design temperatures
    clg_dsgn_sup_air_temp = 55.0 if clg_dsgn_sup_air_temp is None \
        else clg_dsgn_sup_air_temp
    clg_dsgn_sup_air_temp_c = TEMPERATURE.to_unit([clg_dsgn_sup_air_temp], 'C', 'F')[0]
    htg_dsgn_sup_air_temp = 60.0 if htg_dsgn_sup_air_temp is None \
        else htg_dsgn_sup_air_temp
    htg_dsgn_sup_air_temp_c = TEMPERATURE.to_unit([htg_dsgn_sup_air_temp], 'C', 'F')[0]

    # modify system sizing properties
    sizing_system = air_loop.sizingSystem()
    sizing_system.setTypeofLoadtoSizeOn('VentilationRequirement')
    sizing_system.setAllOutdoorAirinCooling(True)
    sizing_system.setAllOutdoorAirinHeating(True)
    # set minimum airflow ratio to 1.0 to avoid under-sizing heating coil
    if model.version() < openstudio.VersionString('2.7.0'):
        sizing_system.setMinimumSystemAirFlowRatio(1.0)
    else:
        sizing_system.setCentralHeatingMaximumSystemAirFlowRatio(1.0)

    sizing_system.setSizingOption('Coincident')
    sizing_system.setCentralCoolingDesignSupplyAirTemperature(clg_dsgn_sup_air_temp_c)
    sizing_system.setCentralHeatingDesignSupplyAirTemperature(htg_dsgn_sup_air_temp_c)

    # create supply fan
    supply_fan = create_fan_by_name(
        model, 'Constant_DOAS_Fan', fan_name='DOAS Supply Fan',
        end_use_subcategory='DOAS Fans')
    supply_fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
    if fan_maximum_flow_rate is not None:
        fan_max_fr = FLOW_RATE.to_unit([fan_maximum_flow_rate], 'm3/s', 'cfm')[0]
        supply_fan.setMaximumFlowRate(fan_max_fr)
    supply_fan.addToNode(air_loop.supplyInletNode())

    # create heating coil
    if hot_water_loop is None:
        # electric backup heating coil
        create_coil_heating_electric(model, air_loop_node=air_loop.supplyInletNode(),
                                     name='{} Backup Htg Coil'.format(loop_name))
        # heat pump coil
        create_coil_heating_dx_single_speed(model, air_loop_node=air_loop.supplyInletNode(),
                                            name='{} Htg Coil'.format(loop_name))
    else:
        create_coil_heating_water(
            model, hot_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Htg Coil'.format(loop_name), controller_convergence_tolerance=0.0001)

    # create cooling coil
    if chilled_water_loop is None:
        create_coil_cooling_dx_two_speed(
            model, air_loop_node=air_loop.supplyInletNode(),
            name='{} 2spd DX Clg Coil'.format(loop_name), type='OS default')
    else:
        create_coil_cooling_water(
            model, chilled_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Clg Coil'.format(loop_name))

    # minimum outdoor air schedule
    if min_oa_sch is None:
        min_oa_sch = model.alwaysOnDiscreteSchedule()
    else:
        min_oa_sch = model_add_schedule(model, min_oa_sch)

    # minimum outdoor air fraction schedule
    if min_frac_oa_sch is None:
        min_frac_oa_sch = model.alwaysOnDiscreteSchedule()
    else:
        min_frac_oa_sch = model_add_schedule(model, min_frac_oa_sch)

    # create controller outdoor air
    controller_oa = openstudio_model.ControllerOutdoorAir(model)
    controller_oa.setName('{} OA Controller'.format(loop_name))
    controller_oa.setEconomizerControlType(econo_ctrl_mthd)
    controller_oa.setMinimumLimitType('FixedMinimum')
    controller_oa.autosizeMinimumOutdoorAirFlowRate()
    controller_oa.setMinimumOutdoorAirSchedule(min_oa_sch)
    controller_oa.setMinimumFractionofOutdoorAirSchedule(min_frac_oa_sch)
    controller_oa.resetEconomizerMaximumLimitDryBulbTemperature()
    controller_oa.resetEconomizerMaximumLimitEnthalpy()
    controller_oa.resetMaximumFractionofOutdoorAirSchedule()
    controller_oa.resetEconomizerMinimumLimitDryBulbTemperature()
    controller_oa.setHeatRecoveryBypassControlType('BypassWhenWithinEconomizerLimits')

    # create outdoor air system
    oa_system = openstudio_model.AirLoopHVACOutdoorAirSystem(model, controller_oa)
    oa_system.setName('{} OA System'.format(loop_name))
    oa_system.addToNode(air_loop.supplyInletNode())

    # create a setpoint manager
    sat_oa_reset = openstudio_model.SetpointManagerOutdoorAirReset(model)
    sat_oa_reset.setName('{} SAT Reset'.format(loop_name))
    sat_oa_reset.setControlVariable('Temperature')
    sat_oa_reset.setSetpointatOutdoorLowTemperature(htg_dsgn_sup_air_temp_c)
    sat_oa_reset.setOutdoorLowTemperature(TEMPERATURE.to_unit([60.0], 'C', 'F')[0])
    sat_oa_reset.setSetpointatOutdoorHighTemperature(clg_dsgn_sup_air_temp_c)
    sat_oa_reset.setOutdoorHighTemperature(TEMPERATURE.to_unit([70.0], 'C', 'F')[0])
    sat_oa_reset.addToNode(air_loop.supplyOutletNode())

    # set air loop availability controls and night cycle manager, after oa system added
    air_loop.setAvailabilitySchedule(hvac_op_sch)
    air_loop.setNightCycleControlType('CycleOnAny')

    # add energy recovery if requested
    if energy_recovery:
        # create the ERV and set its properties
        erv = create_hx_air_to_air_sensible_and_latent(
            model, name='{} ERV HX'.format(loop_name),
            type="Rotary", economizer_lockout=True,
            sensible_heating_100_eff=0.76, latent_heating_100_eff=0.68,
            sensible_cooling_100_eff=0.76, latent_cooling_100_eff=0.68)
        erv.addToNode(oa_system.outboardOANode().get())

        # increase fan static pressure to account for ERV
        erv_pressure_rise = PRESSURE.to_unit([1.0], 'Pa', 'inH2O')[0]
        new_pressure_rise = supply_fan.pressureRise() + erv_pressure_rise
        supply_fan.setPressureRise(new_pressure_rise)

    # add thermal zones to airloop
    for zone in thermal_zones:
        # make an air terminal for the zone
        if model.version() < openstudio.VersionString('2.7.0'):
            air_terminal = openstudio_model.AirTerminalSingleDuctUncontrolled(
                model, model.alwaysOnDiscreteSchedule())
        else:
            air_terminal = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                model, model.alwaysOnDiscreteSchedule())
        air_terminal.setName('{} Air Terminal'.format(zone.nameString()))

        # attach new terminal to the zone and to the airloop
        air_loop.multiAddBranchForZone(zone, air_terminal.to_HVACComponent().get())

        # DOAS sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setAccountforDedicatedOutdoorAirSystem(True)
        sizing_zone.setDedicatedOutdoorAirSystemControlStrategy('ColdSupplyAir')
        sizing_zone.setDedicatedOutdoorAirLowSetpointTemperatureforDesign(
            clg_dsgn_sup_air_temp_c)
        sizing_zone.setDedicatedOutdoorAirHighSetpointTemperatureforDesign(
            htg_dsgn_sup_air_temp_c)

    return air_loop


def model_add_doas(
        model, thermal_zones, system_name=None, doas_type='DOASCV',
        hot_water_loop=None, chilled_water_loop=None, hvac_op_sch=None,
        min_oa_sch=None, min_frac_oa_sch=None, fan_maximum_flow_rate=None,
        econo_ctrl_mthd='NoEconomizer', include_exhaust_fan=True,
        demand_control_ventilation=False, doas_control_strategy='NeutralSupplyAir',
        clg_dsgn_sup_air_temp=60.0, htg_dsgn_sup_air_temp=70.0, energy_recovery=False):
    """Creates a DOAS system with terminal units for each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        doas_type: [String] DOASCV or DOASVAV, determines whether the DOAS is
            operated at scheduled, constant flow rate, or airflow is variable to
            allow for economizing or demand controlled ventilation.
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to connect
            to heating and zone fan coils.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop to
            connect to cooling coil.
        hvac_op_sch: [String] name of the HVAC operation schedule, default is
            always on.
        min_oa_sch: [String] name of the minimum outdoor air schedule, default is
            always on.
        min_frac_oa_sch: [String] name of the minimum fraction of outdoor air
            schedule, default is always on.
        fan_maximum_flow_rate: [Double] fan maximum flow rate in cfm, default
            is autosize.
        econo_ctrl_mthd: [String] economizer control type, default is Fixed Dry Bulb.
            If enabled, the DOAS will be sized for twice the ventilation minimum
            to allow economizing.
        include_exhaust_fan: [Boolean] if true, include an exhaust fan.
        demand_control_ventilation: [Boolean] if true, include demand controlled
            ventilation controls and variable volume fans.
        doas_control_strategy: [String] DOAS control strategy. Valid options
            include NeutralSupplyAir and ColdSupplyAir.
        clg_dsgn_sup_air_temp: [Double] design cooling supply air temperature in
            degrees Fahrenheit, default 65F.
        htg_dsgn_sup_air_temp: [Double] design heating supply air temperature in
            degrees Fahrenheit, default 75F.
        energy_recovery: [Boolean] if true, an ERV will be added to the system.
    """
    # Check the total OA requirement for all zones on the system
    tot_oa_req = 0
    for zone in thermal_zones:
        tot_oa_req += thermal_zone_get_outdoor_airflow_rate(zone)
        if tot_oa_req > 0:
            break

    # If the total OA requirement is zero do not add the DOAS system
    # because the simulations will fail
    if tot_oa_req == 0:
        return None

    # create a DOAS air loop
    air_loop = openstudio_model.AirLoopHVAC(model)
    system_name = '{} Zone DOAS'.format(len(thermal_zones)) \
        if system_name is None else system_name
    air_loop.setName(system_name)
    loop_name = air_loop.nameString()

    # set availability schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # DOAS design temperatures
    clg_dsgn_sup_air_temp = 60.0 if clg_dsgn_sup_air_temp is None \
        else clg_dsgn_sup_air_temp
    clg_dsgn_sup_air_temp_c = TEMPERATURE.to_unit([clg_dsgn_sup_air_temp], 'C', 'F')[0]
    htg_dsgn_sup_air_temp = 70.0 if htg_dsgn_sup_air_temp is None \
        else htg_dsgn_sup_air_temp
    htg_dsgn_sup_air_temp_c = TEMPERATURE.to_unit([htg_dsgn_sup_air_temp], 'C', 'F')[0]

    # modify system sizing properties
    sizing_system = air_loop.sizingSystem()
    sizing_system.setTypeofLoadtoSizeOn('VentilationRequirement')
    sizing_system.setAllOutdoorAirinCooling(True)
    sizing_system.setAllOutdoorAirinHeating(True)
    # set minimum airflow ratio to 1.0 to avoid under-sizing heating coil
    if model.version() < openstudio.VersionString('2.7.0'):
        sizing_system.setMinimumSystemAirFlowRatio(1.0)
    else:
        sizing_system.setCentralHeatingMaximumSystemAirFlowRatio(1.0)

    sizing_system.setSizingOption('Coincident')
    sizing_system.setCentralCoolingDesignSupplyAirTemperature(clg_dsgn_sup_air_temp_c)
    sizing_system.setCentralHeatingDesignSupplyAirTemperature(htg_dsgn_sup_air_temp_c)

    if doas_type == 'DOASCV':
        supply_fan = create_fan_by_name(model, 'Constant_DOAS_Fan',
                                        fan_name='DOAS Supply Fan',
                                        end_use_subcategory='DOAS Fans')
    else:  # DOASVAV
        supply_fan = create_fan_by_name(model, 'Variable_DOAS_Fan',
                                        fan_name='DOAS Supply Fan',
                                        end_use_subcategory='DOAS Fans')

    supply_fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
    if fan_maximum_flow_rate is not None:
        fan_max_fr = FLOW_RATE.to_unit([fan_maximum_flow_rate], 'm3/s', 'cfm')[0]
        supply_fan.setMaximumFlowRate(fan_max_fr)
    supply_fan.addToNode(air_loop.supplyInletNode())

    # create heating coil
    if hot_water_loop is None:
        # electric backup heating coil
        create_coil_heating_electric(model, air_loop_node=air_loop.supplyInletNode(),
                                     name='{} Backup Htg Coil'.format(loop_name))
        # heat pump coil
        create_coil_heating_dx_single_speed(model, air_loop_node=air_loop.supplyInletNode(),
                                            name='{} Htg Coil'.format(loop_name))
    else:
        create_coil_heating_water(
            model, hot_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Htg Coil'.format(loop_name),
            controller_convergence_tolerance=0.0001)

    # create cooling coil
    if chilled_water_loop is None:
        create_coil_cooling_dx_two_speed(
            model, air_loop_node=air_loop.supplyInletNode(),
            name='{} 2spd DX Clg Coil'.format(loop_name), type='OS default')
    else:
        create_coil_cooling_water(
            model, chilled_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Clg Coil'.format(loop_name))

    # minimum outdoor air schedule
    if min_oa_sch is None:
        min_oa_sch = model_add_schedule(model, min_oa_sch)

    # minimum outdoor air fraction schedule
    if min_frac_oa_sch is None:
        min_frac_oa_sch = model.alwaysOnDiscreteSchedule()
    else:
        min_frac_oa_sch = model_add_schedule(model, min_frac_oa_sch)

    # create controller outdoor air
    controller_oa = openstudio_model.ControllerOutdoorAir(model)
    controller_oa.setName('{} Outdoor Air Controller'.format(loop_name))
    controller_oa.setEconomizerControlType(econo_ctrl_mthd)
    controller_oa.setMinimumLimitType('FixedMinimum')
    controller_oa.autosizeMinimumOutdoorAirFlowRate()
    controller_oa.setMinimumOutdoorAirSchedule(min_oa_sch)
    if min_oa_sch is not None:
        controller_oa.setMinimumFractionofOutdoorAirSchedule(min_frac_oa_sch)
    controller_oa.resetEconomizerMinimumLimitDryBulbTemperature()
    controller_oa.resetEconomizerMaximumLimitDryBulbTemperature()
    controller_oa.resetEconomizerMaximumLimitEnthalpy()
    controller_oa.resetMaximumFractionofOutdoorAirSchedule()
    controller_oa.setHeatRecoveryBypassControlType('BypassWhenWithinEconomizerLimits')
    controller_mech_vent = controller_oa.controllerMechanicalVentilation()
    controller_mech_vent.setName('{} Mechanical Ventilation Controller'.format(loop_name))
    if demand_control_ventilation:
        controller_mech_vent.setDemandControlledVentilation(True)
    controller_mech_vent.setSystemOutdoorAirMethod('ZoneSum')

    # create outdoor air system
    oa_system = openstudio_model.AirLoopHVACOutdoorAirSystem(model, controller_oa)
    oa_system.setName('{} OA System'.format(loop_name))
    oa_system.addToNode(air_loop.supplyInletNode())

    # create an exhaust fan
    if include_exhaust_fan:
        if doas_type == 'DOASCV':
            exhaust_fan = create_fan_by_name(model, 'Constant_DOAS_Fan',
                                             fan_name='DOAS Exhaust Fan',
                                             end_use_subcategory='DOAS Fans')
        else:  # 'DOASVAV'
            exhaust_fan = create_fan_by_name(model, 'Variable_DOAS_Fan',
                                             fan_name='DOAS Exhaust Fan',
                                             end_use_subcategory='DOAS Fans')
        # set pressure rise 1.0 inH2O lower than supply fan, 1.0 inH2O minimum
        in_h20 = PRESSURE.to_unit([1.0], 'Pa', 'inH2O')[0]
        exhaust_fan_pressure_rise = supply_fan.pressureRise() - in_h20
        if exhaust_fan_pressure_rise < in_h20:
            exhaust_fan_pressure_rise = in_h20
        exhaust_fan.setPressureRise(exhaust_fan_pressure_rise)
        exhaust_fan.addToNode(air_loop.supplyInletNode())

    # create a setpoint manager
    sat_oa_reset = openstudio_model.SetpointManagerOutdoorAirReset(model)
    sat_oa_reset.setName('{} SAT Reset'.format(loop_name))
    sat_oa_reset.setControlVariable('Temperature')
    sat_oa_reset.setSetpointatOutdoorLowTemperature(htg_dsgn_sup_air_temp_c)
    sat_oa_reset.setOutdoorLowTemperature(TEMPERATURE.to_unit([55.0], 'C', 'F')[0])
    sat_oa_reset.setSetpointatOutdoorHighTemperature(clg_dsgn_sup_air_temp_c)
    sat_oa_reset.setOutdoorHighTemperature(TEMPERATURE.to_unit([70.0], 'C', 'F')[0])
    sat_oa_reset.addToNode(air_loop.supplyOutletNode())

    # set air loop availability controls and night cycle manager, after oa system added
    air_loop.setAvailabilitySchedule(hvac_op_sch)
    air_loop.setNightCycleControlType('CycleOnAnyZoneFansOnly')

    # add energy recovery if requested
    if energy_recovery:
        # create the ERV and set its properties
        erv = create_hx_air_to_air_sensible_and_latent(
            model, name='{} ERV HX'.format(loop_name),
            type="Rotary", economizer_lockout=True,
            sensible_heating_100_eff=0.76, latent_heating_100_eff=0.68,
            sensible_cooling_100_eff=0.76, latent_cooling_100_eff=0.68,)
        erv.addToNode(oa_system.outboardOANode().get())

        # increase fan static pressure to account for ERV
        erv_pressure_rise = PRESSURE.to_unit([1.0], 'Pa', 'inH2O')[0]
        new_pressure_rise = supply_fan.pressureRise() + erv_pressure_rise
        supply_fan.setPressureRise(new_pressure_rise)

    # add thermal zones to airloop
    for zone in thermal_zones:
        # skip zones with no outdoor air flow rate
        if thermal_zone_get_outdoor_airflow_rate(zone) == 0:
            continue
        zone_name = zone.nameString()

        # make an air terminal for the zone
        if doas_type == 'DOASCV':
            if model.version() < openstudio.VersionString('2.7.0'):
                air_terminal = openstudio_model.AirTerminalSingleDuctUncontrolled(
                    model, model.alwaysOnDiscreteSchedule())
            else:
                air_terminal = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                    model, model.alwaysOnDiscreteSchedule())
        elif doas_type == 'DOASVAVReheat':
            # Reheat coil
            if hot_water_loop is None:
                rht_coil = create_coil_heating_electric(
                    model, name='{} Electric Reheat Coil'.format(zone_name))
            else:
                rht_coil = create_coil_heating_water(
                    model, hot_water_loop, name='{} Reheat Coil'.format(zone_name))

            # VAV reheat terminal
            air_terminal = openstudio_model.AirTerminalSingleDuctVAVReheat(
                model, model.alwaysOnDiscreteSchedule(), rht_coil)
            if model.version() < openstudio.VersionString('3.0.1'):
                air_terminal.setZoneMinimumAirFlowMethod('Constant')
            else:
                air_terminal.setZoneMinimumAirFlowInputMethod('Constant')
            if demand_control_ventilation:
                air_terminal.setControlForOutdoorAir(True)
        else:  # DOASVAV
            air_terminal = openstudio_model.AirTerminalSingleDuctVAVNoReheat(
                model, model.alwaysOnDiscreteSchedule())
            if model.version() < openstudio.VersionString('3.0.1'):
                air_terminal.setZoneMinimumAirFlowMethod('Constant')
            else:
                air_terminal.setZoneMinimumAirFlowInputMethod('Constant')
            air_terminal.setConstantMinimumAirFlowFraction(0.1)
            if demand_control_ventilation:
                air_terminal.setControlForOutdoorAir(True)
        air_terminal.setName('{} Air Terminal'.format(zone_name))

        # attach new terminal to the zone and to the airloop
        air_loop.multiAddBranchForZone(zone, air_terminal.to_HVACComponent().get())

        # ensure the DOAS takes priority, so ventilation load is included when
        # treated by other zonal systems
        zone.setCoolingPriority(air_terminal, 1)
        zone.setHeatingPriority(air_terminal, 1)

        # set the cooling and heating fraction to zero so that if DCV is enabled,
        # the system will lower the ventilation rate rather than trying to meet
        # the heating or cooling load.
        if model.version() < openstudio.VersionString('2.8.0'):
            if demand_control_ventilation:
                msg = 'Unable to add DOAS with DCV to model because the ' \
                    'setSequentialCoolingFraction method is not available in ' \
                    'OpenStudio versions less than 2.8.0.'
                print(msg)
        else:
            zone.setSequentialCoolingFraction(air_terminal, 0.0)
            zone.setSequentialHeatingFraction(air_terminal, 0.0)
            # if economizing, override to meet cooling load first with doas supply
            if econo_ctrl_mthd != 'NoEconomizer':
                zone.setSequentialCoolingFraction(air_terminal, 1.0)

        # DOAS sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setAccountforDedicatedOutdoorAirSystem(True)
        sizing_zone.setDedicatedOutdoorAirSystemControlStrategy(doas_control_strategy)
        sizing_zone.setDedicatedOutdoorAirLowSetpointTemperatureforDesign(clg_dsgn_sup_air_temp_c)
        sizing_zone.setDedicatedOutdoorAirHighSetpointTemperatureforDesign(htg_dsgn_sup_air_temp_c)

    return air_loop


def model_add_vav_reheat(
        model, thermal_zones, system_name=None, return_plenum=None, heating_type=None,
        reheat_type=None, hot_water_loop=None, chilled_water_loop=None,
        hvac_op_sch=None, oa_damper_sch=None,
        fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0,
        min_sys_airflow_ratio=0.3, vav_sizing_option='Coincident', econo_ctrl_mthd=None):
    """Creates a VAV system and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        return_plenum: [OpenStudio::Model::ThermalZone] the zone to attach as
            the supply plenum, or None, in which case no return plenum will be used.
        heating_type: [String] main heating coil fuel type. valid choices are
            NaturalGas, Gas, Electricity, HeatPump, DistrictHeating,
            DistrictHeatingWater, DistrictHeatingSteam, or None (defaults to NaturalGas).
        reheat_type: [String] valid options are NaturalGas, Gas, Electricity
            Water, None (no heat).
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to connect
            heating and reheat coils to.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop
            to connect cooling coil to.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in
            which case will be defaulted to always on.
        oa_damper_sch: [String] name of the oa damper schedule, or None in
            which case will be defaulted to always open.
        fan_efficiency: [Double] fan total efficiency, including motor and impeller.
        fan_motor_efficiency: [Double] fan motor efficiency.
        fan_pressure_rise: [Double] fan pressure rise, inH2O.
        min_sys_airflow_ratio: [Double] minimum system airflow ratio.
        vav_sizing_option: [String] air system sizing option, Coincident or NonCoincident.
        econo_ctrl_mthd: [String] economizer control type.
    """
    # create air handler
    air_loop = openstudio_model.AirLoopHVAC(model)
    system_name = '{} Zone VAV'.format(len(thermal_zones)) \
        if system_name is None else system_name
    air_loop.setName(system_name)
    loop_name = air_loop.nameString()

    # set availability schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # oa damper schedule
    if oa_damper_sch is not None:
        oa_damper_sch = model_add_schedule(model, oa_damper_sch)

    # default design temperatures and settings used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()
    sizing_system = adjust_sizing_system(air_loop, dsgn_temps)
    if min_sys_airflow_ratio is not None:
        if model.version() < openstudio.VersionString('2.7.0'):
            sizing_system.setMinimumSystemAirFlowRatio(min_sys_airflow_ratio)
        else:
            sizing_system.setCentralHeatingMaximumSystemAirFlowRatio(min_sys_airflow_ratio)
    if vav_sizing_option is not None:
        sizing_system.setSizingOption(vav_sizing_option)
    if hot_water_loop is not None:
        hw_temp_c = hot_water_loop.sizingPlant().designLoopExitTemperature()
        hw_delta_t_k = hot_water_loop.sizingPlant().loopDesignTemperatureDifference()

    # air handler controls
    sa_temp_sch = create_constant_schedule_ruleset(
        model, dsgn_temps['clg_dsgn_sup_air_temp_c'],
        name='Supply Air Temp - {}F'.format(dsgn_temps['clg_dsgn_sup_air_temp_f']),
        schedule_type_limit='Temperature')
    sa_stpt_manager = openstudio_model.SetpointManagerScheduled(model, sa_temp_sch)
    sa_stpt_manager.setName('{} Supply Air Setpoint Manager'.format(loop_name))
    sa_stpt_manager.addToNode(air_loop.supplyOutletNode())

    # create fan
    fan = create_fan_by_name(
        model, 'VAV_System_Fan', fan_name='{} Fan'.format(loop_name),
        fan_efficiency=fan_efficiency, pressure_rise=fan_pressure_rise,
        motor_efficiency=fan_motor_efficiency, end_use_subcategory='VAV System Fans')
    fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
    fan.addToNode(air_loop.supplyInletNode())

    # create heating coil
    if hot_water_loop is None:
        if heating_type == 'Electricity':
            create_coil_heating_electric(
                model, air_loop_node=air_loop.supplyInletNode(),
                name='{} Main Electric Htg Coil'.format(loop_name))
        else:  # default to NaturalGas
            create_coil_heating_gas(
                model, air_loop_node=air_loop.supplyInletNode(),
                name='{} Main Gas Htg Coil'.format(loop_name))
    else:
        create_coil_heating_water(
            model, hot_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Main Htg Coil'.format(loop_name),
            rated_inlet_water_temperature=hw_temp_c,
            rated_outlet_water_temperature=hw_temp_c - hw_delta_t_k,
            rated_inlet_air_temperature=dsgn_temps['prehtg_dsgn_sup_air_temp_c'],
            rated_outlet_air_temperature=dsgn_temps['htg_dsgn_sup_air_temp_c'])

    # create cooling coil
    if chilled_water_loop is None:
        create_coil_cooling_dx_two_speed(model, air_loop_node=air_loop.supplyInletNode(),
                                         name='{} 2spd DX Clg Coil'.format(loop_name),
                                         type='OS default')
    else:
        create_coil_cooling_water(model, chilled_water_loop,
                                  air_loop_node=air_loop.supplyInletNode(),
                                  name='{} Clg Coil'.format(loop_name))

    # outdoor air intake system
    oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
    oa_intake_controller.setName('{} OA Controller'.format(loop_name))
    oa_intake_controller.setMinimumLimitType('FixedMinimum')
    oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
    oa_intake_controller.resetMaximumFractionofOutdoorAirSchedule()
    oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
    if econo_ctrl_mthd is not None:
        oa_intake_controller.setEconomizerControlType(econo_ctrl_mthd)
    if oa_damper_sch is not None:
        oa_intake_controller.setMinimumOutdoorAirSchedule(oa_damper_sch)
    controller_mv = oa_intake_controller.controllerMechanicalVentilation()
    controller_mv.setName('{} Vent Controller'.format(loop_name))
    controller_mv.setSystemOutdoorAirMethod('ZoneSum')
    oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(model, oa_intake_controller)
    oa_intake.setName('{} OA System'.format(loop_name))
    oa_intake.addToNode(air_loop.supplyInletNode())

    # set air loop availability controls and night cycle manager, after oa system added
    air_loop.setAvailabilitySchedule(hvac_op_sch)
    air_loop.setNightCycleControlType('CycleOnAny')

    if model.version() < openstudio.VersionString('3.5.0'):
        avail_mgr = air_loop.availabilityManager()
        if avail_mgr.is_initialized():
            avail_mgr = avail_mgr.get()
        else:
            avail_mgr = None
    else:
        avail_mgr = air_loop.availabilityManagers()[0]

    if avail_mgr is not None and \
            avail_mgr.to_AvailabilityManagerNightCycle().is_initialized():
        avail_mgr = avail_mgr.to_AvailabilityManagerNightCycle().get()
        avail_mgr.setCyclingRunTime(1800)

    # hook the VAV system to each zone
    for zone in thermal_zones:
        # create reheat coil
        zone_name = zone.nameString()
        if reheat_type in ('NaturalGas', 'Gas'):
            rht_coil = create_coil_heating_gas(
                model, name='{} Gas Reheat Coil'.format(zone_name))
        elif reheat_type == 'Electricity':
            rht_coil = create_coil_heating_electric(
                model, name='{} Electric Reheat Coil'.format(zone_name))
        elif reheat_type == 'Water':
            rht_coil = create_coil_heating_water(
                model, hot_water_loop, name='{} Reheat Coil'.format(zone_name),
                rated_inlet_water_temperature=hw_temp_c,
                rated_outlet_water_temperature=(hw_temp_c - hw_delta_t_k),
                rated_inlet_air_temperature=dsgn_temps['htg_dsgn_sup_air_temp_c'],
                rated_outlet_air_temperature=dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        else:
            pass  # no reheat

        # set zone reheat temperatures depending on reheat
        if reheat_type in ('NaturalGas', 'Gas', 'Electricity', 'Water'):
            # create vav terminal
            terminal = openstudio_model.AirTerminalSingleDuctVAVReheat(
                model, model.alwaysOnDiscreteSchedule(), rht_coil)
            terminal.setName('{} VAV Terminal'.format(zone_name))
            if model.version() < openstudio.VersionString('3.0.1'):
                terminal.setZoneMinimumAirFlowMethod('Constant')
            else:
                terminal.setZoneMinimumAirFlowInputMethod('Constant')
            # default to single maximum control logic
            terminal.setDamperHeatingAction('Normal')
            terminal.setMaximumReheatAirTemperature(
                dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
            air_loop.multiAddBranchForZone(zone, terminal.to_HVACComponent().get())
            # air_terminal_single_duct_vav_reheat_apply_initial_prototype_damper_position
            min_damper_position = 0.3
            terminal.setConstantMinimumAirFlowFraction(min_damper_position)
            # zone sizing
            sizing_zone = zone.sizingZone()
            sizing_zone.setCoolingDesignAirFlowMethod('DesignDayWithLimit')
            sizing_zone.setHeatingDesignAirFlowMethod('DesignDay')
            sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
                dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
            sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
                dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        else:  # no reheat
            # create vav terminal
            terminal = openstudio_model.AirTerminalSingleDuctVAVNoReheat(
                model, model.alwaysOnDiscreteSchedule())
            terminal.setName('{} VAV Terminal'.format(zone_name))
            if model.version() < openstudio.VersionString('3.0.1'):
                terminal.setZoneMinimumAirFlowMethod('Constant')
            else:
                terminal.setZoneMinimumAirFlowInputMethod('Constant')
            air_loop.multiAddBranchForZone(zone, terminal.to_HVACComponent().get())
            # air_terminal_single_duct_vav_reheat_apply_initial_prototype_damper_position
            min_damper_position = 0.3
            terminal.setConstantMinimumAirFlowFraction(min_damper_position)
            # zone sizing
            sizing_zone = zone.sizingZone()
            sizing_zone.setCoolingDesignAirFlowMethod('DesignDayWithLimit')
            sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
                dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        if return_plenum is not None:
            zone.setReturnPlenum(return_plenum)

    return air_loop


def model_add_vav_pfp_boxes(
        model, thermal_zones, system_name=None, chilled_water_loop=None,
        hvac_op_sch=None, oa_damper_sch=None,
        fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0):
    """Creates a VAV system with parallel fan powered boxes and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop to
            connect to the cooling coil.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in which
            case will be defaulted to always on.
        oa_damper_sch: [String] name of the oa damper schedule or None in which
            case will be defaulted to always open.
        fan_efficiency: [Double] fan total efficiency, including motor and impeller.
        fan_motor_efficiency: [Double] fan motor efficiency.
        fan_pressure_rise: [Double] fan pressure rise, inH2O.
    """
    # create air handler
    air_loop = openstudio_model.AirLoopHVAC(model)
    system_name = '{} Zone VAV with PFP Boxes and Reheat'.format(len(thermal_zones)) \
        if system_name is None else system_name
    air_loop.setName(system_name)
    loop_name = air_loop.nameString()

    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # oa damper schedule
    oa_damper_sch = model_add_schedule(model, oa_damper_sch)

    # default design temperatures and settings used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()
    adjust_sizing_system(air_loop, dsgn_temps)

    # air handler controls
    sa_temp_sch = create_constant_schedule_ruleset(
        model, dsgn_temps['clg_dsgn_sup_air_temp_c'],
        name='Supply Air Temp - {}F'.format(dsgn_temps['clg_dsgn_sup_air_temp_f']),
        schedule_type_limit='Temperature')
    sa_stpt_manager = openstudio_model.SetpointManagerScheduled(model, sa_temp_sch)
    sa_stpt_manager.setName('{} Supply Air Setpoint Manager'.format(loop_name))
    sa_stpt_manager.addToNode(air_loop.supplyOutletNode())

    # create fan
    fan = create_fan_by_name(
        model, 'VAV_System_Fan', fan_name='{} Fan'.format(loop_name),
        fan_efficiency=fan_efficiency, pressure_rise=fan_pressure_rise,
        motor_efficiency=fan_motor_efficiency, end_use_subcategory='VAV System Fans')
    fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
    fan.addToNode(air_loop.supplyInletNode())

    # create heating coil
    create_coil_heating_electric(
        model, air_loop_node=air_loop.supplyInletNode(),
        name='{} Htg Coil'.format(loop_name))

    # create cooling coil
    create_coil_cooling_water(
        model, chilled_water_loop, air_loop_node=air_loop.supplyInletNode(),
        name='{} Clg Coil'.format(loop_name))

    # create outdoor air intake system
    oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
    oa_intake_controller.setName('{} OA Controller'.format(loop_name))
    oa_intake_controller.setMinimumLimitType('FixedMinimum')
    oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
    oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
    controller_mv = oa_intake_controller.controllerMechanicalVentilation()
    controller_mv.setName('{} Vent Controller'.format(loop_name))
    controller_mv.setSystemOutdoorAirMethod('ZoneSum')
    oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(model, oa_intake_controller)
    oa_intake.setName('{} OA System'.format(loop_name))
    oa_intake.addToNode(air_loop.supplyInletNode())

    # set air loop availability controls and night cycle manager, after oa system added
    air_loop.setAvailabilitySchedule(hvac_op_sch)
    air_loop.setNightCycleControlType('CycleOnAny')

    # attach the VAV system to each zone
    for zone in thermal_zones:
        # create reheat coil
        zone_name = zone.nameString()
        rht_coil = create_coil_heating_electric(
            model, name='{} Electric Reheat Coil'.format(zone_name))

        # create terminal fan
        pfp_fan = create_fan_by_name(
            model, 'PFP_Fan', fan_name='{} PFP Term Fan'.format(zone_name))
        pfp_fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        # create parallel fan powered terminal
        pfp_terminal = openstudio_model.AirTerminalSingleDuctParallelPIUReheat(
            model, model.alwaysOnDiscreteSchedule(), pfp_fan, rht_coil)
        pfp_terminal.setName('{} PFP Term'.format(zone_name))
        air_loop.multiAddBranchForZone(zone, pfp_terminal.to_HVACComponent().get())

        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setCoolingDesignAirFlowMethod('DesignDay')
        sizing_zone.setHeatingDesignAirFlowMethod('DesignDay')
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

    return air_loop


def model_add_pvav(
        model, thermal_zones, system_name=None, return_plenum=None,
        hot_water_loop=None, chilled_water_loop=None, heating_type=None,
        electric_reheat=False, hvac_op_sch=None, oa_damper_sch=None,
        econo_ctrl_mthd=None):
    """Creates a packaged VAV system and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        return_plenum: [OpenStudio::Model::ThermalZone] the zone to attach as
            the supply plenum, or None, in which case no return plenum will be used.
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to connect
            heating and reheat coils to. If None, will be electric heat and electric reheat.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop
            to connect cooling coils to. If None, will be DX cooling.
        heating_type: [String] main heating coil fuel type. Valid choices are
            NaturalGas, Electricity, Water, or None (defaults to NaturalGas).
        electric_reheat: [Boolean] if true electric reheat coils, if false the
            reheat coils served by hot_water_loop.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in
            which case will be defaulted to always on.
        oa_damper_sch: [String] name of the oa damper schedule or None in which
            case will be defaulted to always open.
        econo_ctrl_mthd: [String] economizer control type.
    """
    # create air handler
    air_loop = openstudio_model.AirLoopHVAC(model)
    system_name = '{} Zone PVAV'.format(len(thermal_zones)) \
        if system_name is None else system_name
    air_loop.setName(system_name)
    loop_name = air_loop.nameString()

    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # oa damper schedule
    oa_damper_sch = model_add_schedule(model, oa_damper_sch)

    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()
    if hot_water_loop is not None:
        hw_temp_c = hot_water_loop.sizingPlant().designLoopExitTemperature()
        hw_delta_t_k = hot_water_loop.sizingPlant().loopDesignTemperatureDifference()

    # adjusted zone design heating temperature for pvav unless it would cause
    # a temperature higher than reheat water supply temperature
    if hot_water_loop is not None and hw_temp_c < TEMPERATURE.to_unit([140.0], 'C', 'F')[0]:
        dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
        dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
            TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]

    # default design settings used across all air loops
    adjust_sizing_system(air_loop, dsgn_temps)

    # air handler controls
    sa_temp_sch = create_constant_schedule_ruleset(
        model, dsgn_temps['clg_dsgn_sup_air_temp_c'],
        name='Supply Air Temp - {}F'.format(dsgn_temps['clg_dsgn_sup_air_temp_f']),
        schedule_type_limit='Temperature')
    sa_stpt_manager = openstudio_model.SetpointManagerScheduled(model, sa_temp_sch)
    sa_stpt_manager.setName('{} Supply Air Setpoint Manager'.format(loop_name))
    sa_stpt_manager.addToNode(air_loop.supplyOutletNode())

    # create fan
    fan = create_fan_by_name(model, 'VAV_default', fan_name='{} Fan'.format(loop_name))
    fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
    fan.addToNode(air_loop.supplyInletNode())

    # create heating coil
    if hot_water_loop is None:
        if heating_type == 'Electricity':
            create_coil_heating_electric(
                model, air_loop_node=air_loop.supplyInletNode(),
                name='{} Main Electric Htg Coil'.format(loop_name))
        else:  # default to NaturalGas
            create_coil_heating_gas(
                model, air_loop_node=air_loop.supplyInletNode(),
                name='{} Main Gas Htg Coil'.format(loop_name))
    else:
        create_coil_heating_water(
            model, hot_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Main Htg Coil'.format(loop_name),
            rated_inlet_water_temperature=hw_temp_c,
            rated_outlet_water_temperature=hw_temp_c - hw_delta_t_k,
            rated_inlet_air_temperature=dsgn_temps['prehtg_dsgn_sup_air_temp_c'],
            rated_outlet_air_temperature=dsgn_temps['htg_dsgn_sup_air_temp_c'])

    # create cooling coil
    if chilled_water_loop is None:
        create_coil_cooling_dx_two_speed(
            model, air_loop_node=air_loop.supplyInletNode(),
            name='{} 2spd DX Clg Coil'.format(loop_name), type='OS default')
    else:
        create_coil_cooling_water(
            model, chilled_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Clg Coil'.format(loop_name))

    # outdoor air intake system
    oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
    oa_intake_controller.setName('{} OA Controller'.format(loop_name))
    oa_intake_controller.setMinimumLimitType('FixedMinimum')
    oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
    oa_intake_controller.resetMaximumFractionofOutdoorAirSchedule()
    oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
    if econo_ctrl_mthd is not None:
        oa_intake_controller.setEconomizerControlType(econo_ctrl_mthd)
    if oa_damper_sch is not None:
        oa_intake_controller.setMinimumOutdoorAirSchedule(oa_damper_sch)
    controller_mv = oa_intake_controller.controllerMechanicalVentilation()
    controller_mv.setName('{} Mechanical Ventilation Controller'.format(loop_name))
    controller_mv.setSystemOutdoorAirMethod('ZoneSum')
    oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(model, oa_intake_controller)
    oa_intake.setName('{} OA System'.format(loop_name))
    oa_intake.addToNode(air_loop.supplyInletNode())

    # set air loop availability controls and night cycle manager, after oa system added
    air_loop.setAvailabilitySchedule(hvac_op_sch)
    air_loop.setNightCycleControlType('CycleOnAny')

    if model.version() < openstudio.VersionString('3.5.0'):
        avail_mgr = air_loop.availabilityManager()
        if avail_mgr.is_initialized():
            avail_mgr = avail_mgr.get()
        else:
            avail_mgr = None
    else:
        avail_mgr = air_loop.availabilityManagers()[0]

    if avail_mgr is not None and \
            avail_mgr.to_AvailabilityManagerNightCycle().is_initialized():
        avail_mgr = avail_mgr.to_AvailabilityManagerNightCycle().get()
        avail_mgr.setCyclingRunTime(1800)

    # attach the VAV system to each zone
    for zone in thermal_zones:
        zone_name = zone.nameString()
        # create reheat coil
        if electric_reheat or hot_water_loop is None:
            rht_coil = create_coil_heating_electric(
                model, name='{} Electric Reheat Coil'.format(zone_name))
        else:
            rht_coil = create_coil_heating_water(
                model, hot_water_loop, name='{} Reheat Coil'.format(zone_name),
                rated_inlet_water_temperature=hw_temp_c,
                rated_outlet_water_temperature=hw_temp_c - hw_delta_t_k,
                rated_inlet_air_temperature=dsgn_temps['htg_dsgn_sup_air_temp_c'],
                rated_outlet_air_temperature=dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

        # create VAV terminal
        terminal = openstudio_model.AirTerminalSingleDuctVAVReheat(
            model, model.alwaysOnDiscreteSchedule(), rht_coil)
        terminal.setName('{} VAV Terminal'.format(zone_name))
        if model.version() < openstudio.VersionString('3.0.1'):
            terminal.setZoneMinimumAirFlowMethod('Constant')
        else:
            terminal.setZoneMinimumAirFlowInputMethod('Constant')
        # default to single maximum control logic
        terminal.setDamperHeatingAction('Normal')
        terminal.setMaximumReheatAirTemperature(dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        air_loop.multiAddBranchForZone(zone, terminal.to_HVACComponent().get())
        # air_terminal_single_duct_vav_reheat_apply_initial_prototype_damper_position
        min_damper_position = 0.3
        terminal.setConstantMinimumAirFlowFraction(min_damper_position)
        if return_plenum is not None:
            zone.setReturnPlenum(return_plenum)
        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

    return air_loop


def model_add_pvav_pfp_boxes(
        model, thermal_zones, system_name=None, chilled_water_loop=None,
        hvac_op_sch=None, oa_damper_sch=None,
        fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0):
    """Creates a packaged VAV system with parallel fan powered boxes.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop
            to connect cooling coils to. If None, will be DX cooling.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in
            which case will be defaulted to always on.
        oa_damper_sch: [String] name of the oa damper schedule or None in
            which case will be defaulted to always open.
        fan_efficiency: [Double] fan total efficiency, including motor and impeller.
        fan_motor_efficiency: [Double] fan motor efficiency.
        fan_pressure_rise: [Double] fan pressure rise, inH2O.
    """
    # create air handler
    air_loop = openstudio_model.AirLoopHVAC(model)
    system_name = '{} Zone PVAV with PFP Boxes and Reheat'.format(len(thermal_zones)) \
        if system_name is None else system_name
    air_loop.setName(system_name)
    loop_name = air_loop.nameString()

    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # oa damper schedule
    oa_damper_sch = model_add_schedule(model, oa_damper_sch)

    # default design temperatures and settings used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()
    adjust_sizing_system(air_loop, dsgn_temps)

    # air handler controls
    sa_temp_sch = create_constant_schedule_ruleset(
        model, dsgn_temps['clg_dsgn_sup_air_temp_c'],
        name='Supply Air Temp - {}F'.format(dsgn_temps['clg_dsgn_sup_air_temp_f']),
        schedule_type_limit='Temperature')
    sa_stpt_manager = openstudio_model.SetpointManagerScheduled(model, sa_temp_sch)
    sa_stpt_manager.setName('{} Supply Air Setpoint Manager'.format(loop_name))
    sa_stpt_manager.addToNode(air_loop.supplyOutletNode())

    # create fan
    fan = create_fan_by_name(
        model, 'VAV_System_Fan', fan_name='{} Fan'.format(loop_name),
        fan_efficiency=fan_efficiency, pressure_rise=fan_pressure_rise,
        motor_efficiency=fan_motor_efficiency, end_use_subcategory='VAV System Fans')
    fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
    fan.addToNode(air_loop.supplyInletNode())

    # create heating coil
    create_coil_heating_electric(
        model, air_loop_node=air_loop.supplyInletNode(),
        name='{} Main Htg Coil'.format(loop_name))

    # create cooling coil
    if chilled_water_loop is None:
        create_coil_cooling_dx_two_speed(
            model, air_loop_node=air_loop.supplyInletNode(),
            name='{} 2spd DX Clg Coil'.format(loop_name), type='OS default')
    else:
        create_coil_cooling_water(
            model, chilled_water_loop, air_loop_node=air_loop.supplyInletNode(),
            name='{} Clg Coil'.format(loop_name))

    # create outdoor air intake system
    oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
    oa_intake_controller.setName('{} OA Controller'.format(loop_name))
    oa_intake_controller.setMinimumLimitType('FixedMinimum')
    oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
    oa_intake_controller.setMinimumOutdoorAirSchedule(oa_damper_sch)
    oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
    controller_mv = oa_intake_controller.controllerMechanicalVentilation()
    controller_mv.setName('{} Vent Controller'.format(loop_name))
    controller_mv.setSystemOutdoorAirMethod('ZoneSum')

    oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(model, oa_intake_controller)
    oa_intake.setName('{} OA System'.format(loop_name))
    oa_intake.addToNode(air_loop.supplyInletNode())

    # set air loop availability controls and night cycle manager, after oa system added
    air_loop.setAvailabilitySchedule(hvac_op_sch)
    air_loop.setNightCycleControlType('CycleOnAny')

    # attach the VAV system to each zone
    for zone in thermal_zones:
        zone_name = zone.nameString()
        # create electric reheat coil
        rht_coil = create_coil_heating_electric(
            model, name='{} Electric Reheat Coil'.format(zone_name))

        # create terminal fan
        pfp_fan = create_fan_by_name(model, 'PFP_Fan',
                                     fan_name='{} PFP Term Fan'.format(zone_name))
        pfp_fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        # parallel fan powered terminal
        pfp_terminal = openstudio_model.AirTerminalSingleDuctParallelPIUReheat(
            model, model.alwaysOnDiscreteSchedule(), pfp_fan, rht_coil)
        pfp_terminal.setName("#{zone.name} PFP Term")
        air_loop.multiAddBranchForZone(zone, pfp_terminal.to_HVACComponent().get())

        # adjust zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setCoolingDesignAirFlowMethod('DesignDay')
        sizing_zone.setHeatingDesignAirFlowMethod('DesignDay')
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

    return air_loop


def model_add_psz_ac(
        model, thermal_zones, system_name=None, cooling_type='Single Speed DX AC',
        chilled_water_loop=None, hot_water_loop=None, heating_type=None,
        supplemental_heating_type=None, fan_location='DrawThrough',
        fan_type='ConstantVolume', hvac_op_sch=None, oa_damper_sch=None):
    """Creates a PSZ-AC system for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        cooling_type: [String] valid choices are Water, Two Speed DX AC,
            Single Speed DX AC, Single Speed Heat Pump, Water To Air Heat Pump.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop
            to connect cooling coil to, or None.
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to
            connect heating coil to, or None.
        heating_type: [String] valid choices are NaturalGas, Electricity,
            Water, Single Speed Heat Pump, Water To Air Heat Pump, or None (no heat).
        supplemental_heating_type: [String] valid choices are Electricity,
            NaturalGas, None (no heat).
        fan_location: [String] valid choices are BlowThrough, DrawThrough.
        fan_type: [String] valid choices are ConstantVolume, Cycling.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in
            which case will be defaulted to always on.
        oa_damper_sch: [String] name of the oa damper schedule or None in
            which case will be defaulted to always open.
    """
    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # oa damper schedule
    oa_damper_sch = model_add_schedule(model, oa_damper_sch)

    # create a PSZ-AC for each zone
    air_loops = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        air_loop = openstudio_model.AirLoopHVAC(model)
        system_name = '{} PSZ-AC'.format(zone_name) \
            if system_name is None else '{} {}'.format(zone_name, system_name)
        air_loop.setName(system_name)
        loop_name = air_loop.nameString()

        # default design temperatures and settings used across all air loops
        dsgn_temps = standard_design_sizing_temperatures()
        if hot_water_loop is not None:
            hw_temp_c = hot_water_loop.sizingPlant().designLoopExitTemperature()
            hw_delta_t_k = hot_water_loop.sizingPlant().loopDesignTemperatureDifference()

        # adjusted design heating temperature for psz_ac
        dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
        dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
            TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
        dsgn_temps['htg_dsgn_sup_air_temp_f'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_f']
        dsgn_temps['htg_dsgn_sup_air_temp_c'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_c']

        # default design settings used across all air loops
        adjust_sizing_system(air_loop, dsgn_temps, min_sys_airflow_ratio=1.0)

        # air handler controls
        # add a setpoint manager single zone reheat to control the supply air temperature
        setpoint_mgr_single_zone_reheat = \
            openstudio_model.SetpointManagerSingleZoneReheat(model)
        setpoint_mgr_single_zone_reheat.setName(
            '{} Setpoint Manager SZ Reheat'.format(zone_name))
        setpoint_mgr_single_zone_reheat.setControlZone(zone)
        setpoint_mgr_single_zone_reheat.setMinimumSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        setpoint_mgr_single_zone_reheat.setMaximumSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        setpoint_mgr_single_zone_reheat.addToNode(air_loop.supplyOutletNode())

        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

        # create heating coil
        if heating_type in ('NaturalGas', 'Gas'):
            htg_coil = create_coil_heating_gas(
                model, name='{} Gas Htg Coil'.format(loop_name))
        elif heating_type == 'Water':
            if hot_water_loop is None:
                print('No hot water plant loop supplied')
                return None
            htg_coil = create_coil_heating_water(
                model, hot_water_loop, name='{} Water Htg Coil'.format(loop_name),
                rated_inlet_water_temperature=hw_temp_c,
                rated_outlet_water_temperature=hw_temp_c - hw_delta_t_k,
                rated_inlet_air_temperature=dsgn_temps['prehtg_dsgn_sup_air_temp_c'],
                rated_outlet_air_temperature=dsgn_temps['htg_dsgn_sup_air_temp_c'])
        elif heating_type == 'Single Speed Heat Pump':
            htg_coil = create_coil_heating_dx_single_speed(
                model, name='{} HP Htg Coil'.format(zone_name), type='PSZ-AC', cop=3.3)
        elif heating_type == 'Water To Air Heat Pump':
            htg_coil = create_coil_heating_water_to_air_heat_pump_equation_fit(
                model, hot_water_loop,
                name='{} Water-to-Air HP Htg Coil'.format(loop_name))
        elif heating_type in ('Electricity', 'Electric'):
            htg_coil = create_coil_heating_electric(
                model, name='{} Electric Htg Coil'.format(loop_name))
        else:
            # zero-capacity, always-off electric heating coil
            htg_coil = create_coil_heating_electric(
                model, name='{} No Heat'.format(loop_name),
                schedule=model.alwaysOffDiscreteSchedule(),
                nominal_capacity=0.0)

        # create supplemental heating coil
        if supplemental_heating_type in ('Electricity', 'Electric'):
            supplemental_htg_coil = create_coil_heating_electric(
                model, name='{} Electric Backup Htg Coil'.format(loop_name))
        elif supplemental_heating_type in ('NaturalGas', 'Gas'):
            supplemental_htg_coil = create_coil_heating_gas(
                model, name='{} Gas Backup Htg Coil'.format(loop_name))
        else:  # Zero-capacity, always-off electric heating coil
            supplemental_htg_coil = create_coil_heating_electric(
                model, name='{} No Heat'.format(loop_name),
                schedule=model.alwaysOffDiscreteSchedule(), nominal_capacity=0.0)

        # create cooling coil
        if cooling_type == 'Water':
            if chilled_water_loop is None:
                print('No chilled water plant loop supplied')
                return None
            clg_coil = create_coil_cooling_water(
                model, chilled_water_loop, name='{} Water Clg Coil'.format(loop_name))
        elif cooling_type == 'Two Speed DX AC':
            clg_coil = create_coil_cooling_dx_two_speed(
                model, name='{} 2spd DX AC Clg Coil'.format(loop_name))
        elif cooling_type == 'Single Speed DX AC':
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} 1spd DX AC Clg Coil'.format(loop_name), type='PSZ-AC')
        elif cooling_type == 'Single Speed Heat Pump':
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} 1spd DX HP Clg Coil'.format(loop_name), type='Heat Pump')
        elif cooling_type == 'Water To Air Heat Pump':
            if chilled_water_loop is None:
                print('No chilled water plant loop supplied')
                return None
            clg_coil = create_coil_cooling_water_to_air_heat_pump_equation_fit(
                model, chilled_water_loop,
                name='{} Water-to-Air HP Clg Coil'.format(loop_name))
        else:
            clg_coil = None

        # Use a Fan:OnOff in the unitary system object
        if fan_type == 'Cycling':
            fan = create_fan_by_name(model, 'Packaged_RTU_SZ_AC_Cycling_Fan',
                                     fan_name='{} Fan'.format(loop_name))
        elif fan_type == 'ConstantVolume':
            fan = create_fan_by_name(model, 'Packaged_RTU_SZ_AC_CAV_OnOff_Fan',
                                     fan_name='{} Fan'.format(loop_name))
        else:
            raise ValueError('Invalid PSZ-AC fan_type "{}".'.format(fan_type))

        # fan location
        fan_location = 'DrawThrough' if fan_location is None else fan_location
        if fan_location not in ('DrawThrough', 'BlowThrough'):
            msg = 'Invalid fan_location {} for fan {}.'.format(
                fan_location, fan.nameString())
            raise ValueError(msg)

        # construct unitary system object
        unitary_system = openstudio_model.AirLoopHVACUnitarySystem(model)
        if fan is not None:
            unitary_system.setSupplyFan(fan)
        if htg_coil is not None:
            unitary_system.setHeatingCoil(htg_coil)
        if clg_coil is not None:
            unitary_system.setCoolingCoil(clg_coil)
        if supplemental_htg_coil is not None:
            unitary_system.setSupplementalHeatingCoil(supplemental_htg_coil)
        unitary_system.setControllingZoneorThermostatLocation(zone)
        unitary_system.setFanPlacement(fan_location)
        unitary_system.addToNode(air_loop.supplyInletNode())

        # added logic and naming for heat pumps
        if heating_type == 'Water To Air Heat Pump':
            unitary_system.setMaximumOutdoorDryBulbTemperatureforSupplementalHeaterOperation(
                TEMPERATURE.to_unit([40.0], 'C', 'F')[0])
            unitary_system.setName('{} Unitary HP'.format(loop_name))
            unitary_system.setMaximumSupplyAirTemperature(
                dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
            if model.version() < openstudio.VersionString('3.7.0'):
                unitary_system.setSupplyAirFlowRateMethodDuringCoolingOperation(
                    'SupplyAirFlowRate')
                unitary_system.setSupplyAirFlowRateMethodDuringHeatingOperation(
                    'SupplyAirFlowRate')
                unitary_system.setSupplyAirFlowRateMethodWhenNoCoolingorHeatingisRequired(
                    'SupplyAirFlowRate')
            else:
                unitary_system.autosizeSupplyAirFlowRateDuringCoolingOperation()
                unitary_system.autosizeSupplyAirFlowRateDuringHeatingOperation()
                unitary_system.autosizeSupplyAirFlowRateWhenNoCoolingorHeatingisRequired()
        elif heating_type == 'Single Speed Heat Pump':
            unitary_system.setMaximumOutdoorDryBulbTemperatureforSupplementalHeaterOperation(
                TEMPERATURE.to_unit([40.0], 'C', 'F')[0])
            unitary_system.setName('{} Unitary HP'.format(loop_name))
        else:
            unitary_system.setName('{} Unitary AC'.format(loop_name))

        # specify control logic
        unitary_system.setAvailabilitySchedule(hvac_op_sch)
        if fan_type == 'Cycling':
            unitary_system.setSupplyAirFanOperatingModeSchedule(
                model.alwaysOffDiscreteSchedule())
        else:  # constant volume operation
            unitary_system.setSupplyAirFanOperatingModeSchedule(hvac_op_sch)

        # add the OA system
        oa_controller = openstudio_model.ControllerOutdoorAir(model)
        oa_controller.setName('{} OA System Controller'.format(loop_name))
        oa_controller.setMinimumOutdoorAirSchedule(oa_damper_sch)
        oa_controller.autosizeMinimumOutdoorAirFlowRate()
        oa_controller.resetEconomizerMinimumLimitDryBulbTemperature()
        oa_system = openstudio_model.AirLoopHVACOutdoorAirSystem(model, oa_controller)
        oa_system.setName('{} OA System'.format(loop_name))
        oa_system.addToNode(air_loop.supplyInletNode())

        # set air loop availability controls and night cycle manager, after oa system added
        air_loop.setAvailabilitySchedule(hvac_op_sch)
        air_loop.setNightCycleControlType('CycleOnAny')

        if model.version() < openstudio.VersionString('3.5.0'):
            avail_mgr = air_loop.availabilityManager()
            avail_mgr = avail_mgr.get() if avail_mgr.is_initialized() else None
        else:
            avail_mgr = air_loop.availabilityManagers()[0]

        if avail_mgr is not None and \
                avail_mgr.to_AvailabilityManagerNightCycle().is_initialized():
            avail_mgr = avail_mgr.to_AvailabilityManagerNightCycle().get()
            avail_mgr.setCyclingRunTime(1800)

        # create a diffuser and attach the zone/diffuser pair to the air loop
        if model.version() < openstudio.VersionString('2.7.0'):
            diffuser = openstudio_model.AirTerminalSingleDuctUncontrolled(
                model, model.alwaysOnDiscreteSchedule())
        else:
            diffuser = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                model, model.alwaysOnDiscreteSchedule())
        diffuser.setName('{} Diffuser'.format(loop_name))
        air_loop.multiAddBranchForZone(zone, diffuser.to_HVACComponent().get())
        air_loops.append(air_loop)

    return air_loops


def model_add_psz_vav(
        model, thermal_zones, system_name=None, heating_type=None,
        cooling_type='AirCooled', supplemental_heating_type=None, hvac_op_sch=None,
        fan_type='VAV_System_Fan', oa_damper_sch=None, hot_water_loop=None,
        chilled_water_loop=None, minimum_volume_setpoint=None):
    """Creates a packaged single zone VAV system for each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to connect to this system.
        system_name: [String] the name of the system, or None in which case it
            will be defaulted.
        heating_type: [String] valid choices are NaturalGas, Electricity, Water,
            None (no heat).
        supplemental_heating_type: [String] valid choices are Electricity,
            NaturalGas, None (no heat).
        hvac_op_sch: [String] name of the HVAC operation schedule or None in
            which case will be defaulted to always on.
        oa_damper_sch: [String] name of the oa damper schedule or None in which
            case will be defaulted to always open.
    """
    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # oa damper schedule
    oa_damper_sch = model_add_schedule(model, oa_damper_sch)

    # create a PSZ-AC for each zone
    air_loops = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        air_loop = openstudio_model.AirLoopHVAC(model)
        system_name = '{} PSZ-VAV'.format(zone_name) \
            if system_name is None else '{} {}'.format(zone_name, system_name)
        air_loop.setName(system_name)
        loop_name = air_loop.nameString()

        # default design temperatures used across all air loops
        dsgn_temps = standard_design_sizing_temperatures()

        # adjusted zone design heating temperature for psz_vav
        dsgn_temps['htg_dsgn_sup_air_temp_f'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_f']
        dsgn_temps['htg_dsgn_sup_air_temp_c'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_c']

        # default design settings used across all air loops
        adjust_sizing_system(air_loop, dsgn_temps)

        # air handler controls
        # add a setpoint manager single zone reheat to control the supply air temperature
        setpoint_mgr_single_zone_reheat = \
            openstudio_model.SetpointManagerSingleZoneReheat(model)
        setpoint_mgr_single_zone_reheat.setName(
            '{} Setpoint Manager SZ Reheat'.format(zone_name))
        setpoint_mgr_single_zone_reheat.setControlZone(zone)
        setpoint_mgr_single_zone_reheat.setMinimumSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        setpoint_mgr_single_zone_reheat.setMaximumSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        setpoint_mgr_single_zone_reheat.addToNode(air_loop.supplyOutletNode())

        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

        # create fan
        fan = create_fan_by_name(model, fan_type, fan_name='{} Fan'.format(loop_name),
                                 end_use_subcategory='VAV System Fans')
        fan.setAvailabilitySchedule(hvac_op_sch)

        # create heating coil
        if heating_type in ('NaturalGas', 'Gas'):
            htg_coil = create_coil_heating_gas(
                model, name='{} Gas Htg Coil'.format(loop_name))
        elif heating_type in ('Electricity', 'Electric'):
            htg_coil = create_coil_heating_electric(
                model, name='{} Electric Htg Coil'.format(loop_name))
        elif heating_type == 'Water':
            htg_coil = create_coil_heating_water(
                model, hot_water_loop, name='{} Water Htg Coil'.format(loop_name))
        else:  # Zero-capacity, always-off electric heating coil
            htg_coil = create_coil_heating_electric(
                model, name='{} No Heat'.format(loop_name),
                schedule=model.alwaysOffDiscreteSchedule(), nominal_capacity=0.0)

        # create supplemental heating coil
        if supplemental_heating_type in ('Electricity', 'Electric'):
            supplemental_htg_coil = create_coil_heating_electric(
                model, name='{} Electric Backup Htg Coil'.format(loop_name))
        elif supplemental_heating_type in ('NaturalGas', 'Gas'):
            supplemental_htg_coil = create_coil_heating_gas(
                model, name='{} Gas Backup Htg Coil'.format(loop_name))
        else:  # zero-capacity, always-off electric heating coil
            supplemental_htg_coil = create_coil_heating_electric(
                model, name='{} No Backup Heat'.format(loop_name),
                schedule=model.alwaysOffDiscreteSchedule(), nominal_capacity=0.0)

        # create cooling coil
        if cooling_type == 'WaterCooled':
            clg_coil = create_coil_cooling_water(
                model, chilled_water_loop, name='{} Clg Coil'.format(loop_name))
        else:  # AirCooled
            clg_coil = openstudio_model.CoilCoolingDXVariableSpeed(model)
            clg_coil.setName('{} Var spd DX AC Clg Coil'.format(loop_name))
            clg_coil.setBasinHeaterCapacity(10.0)
            clg_coil.setBasinHeaterSetpointTemperature(2.0)
            # first speed level
            clg_spd_1 = openstudio_model.CoilCoolingDXVariableSpeedSpeedData(model)
            clg_coil.addSpeed(clg_spd_1)
            clg_coil.setNominalSpeedLevel(1)

        # wrap coils in a unitary system
        unitary_system = openstudio_model.AirLoopHVACUnitarySystem(model)
        unitary_system.setSupplyFan(fan)
        unitary_system.setHeatingCoil(htg_coil)
        unitary_system.setCoolingCoil(clg_coil)
        unitary_system.setSupplementalHeatingCoil(supplemental_htg_coil)
        unitary_system.setName('{} Unitary PSZ-VAV'.format(zone_name))
        # The following control strategy can lead to "Developer Error: Component sizing incomplete."
        # EnergyPlus severe (not fatal) errors if there is no heating design load
        unitary_system.setControlType('SingleZoneVAV')
        unitary_system.setControllingZoneorThermostatLocation(zone)
        unitary_system.setMaximumSupplyAirTemperature(dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        unitary_system.setFanPlacement('BlowThrough')
        if model.version() < openstudio.VersionString('3.7.0'):
            unitary_system.setSupplyAirFlowRateMethodDuringCoolingOperation(
                'SupplyAirFlowRate')
            unitary_system.setSupplyAirFlowRateMethodDuringHeatingOperation(
                'SupplyAirFlowRate')
            if minimum_volume_setpoint is None:
                unitary_system.setSupplyAirFlowRateMethodWhenNoCoolingorHeatingisRequired(
                    'SupplyAirFlowRate')
            else:
                us = unitary_system
                us.setSupplyAirFlowRateMethodWhenNoCoolingorHeatingisRequired(
                    'FractionOfAutosizedCoolingValue')
                us.setFractionofAutosizedDesignCoolingSupplyAirFlowRateWhenNoCoolingorHeatingisRequired(
                    minimum_volume_setpoint)
        else:
            unitary_system.autosizeSupplyAirFlowRateDuringCoolingOperation()
            unitary_system.autosizeSupplyAirFlowRateDuringHeatingOperation()
            if minimum_volume_setpoint is None:
                unitary_system.autosizeSupplyAirFlowRateWhenNoCoolingorHeatingisRequired()
            else:
                us = unitary_system
                us.setFractionofAutosizedDesignCoolingSupplyAirFlowRateWhenNoCoolingorHeatingisRequired(
                    minimum_volume_setpoint)
        unitary_system.setSupplyAirFanOperatingModeSchedule(model.alwaysOnDiscreteSchedule())
        unitary_system.addToNode(air_loop.supplyInletNode())

        # create outdoor air system
        oa_controller = openstudio_model.ControllerOutdoorAir(model)
        oa_controller.setName('{} OA Sys Controller'.format(loop_name))
        oa_controller.setMinimumOutdoorAirSchedule(oa_damper_sch)
        oa_controller.autosizeMinimumOutdoorAirFlowRate()
        oa_controller.resetEconomizerMinimumLimitDryBulbTemperature()
        oa_controller.setHeatRecoveryBypassControlType('BypassWhenOAFlowGreaterThanMinimum')
        oa_system = openstudio_model.AirLoopHVACOutdoorAirSystem(model, oa_controller)
        oa_system.setName('{} OA System'.format(loop_name))
        oa_system.addToNode(air_loop.supplyInletNode())

        # set air loop availability controls and night cycle manager, after oa system added
        air_loop.setAvailabilitySchedule(hvac_op_sch)
        air_loop.setNightCycleControlType('CycleOnAny')

        # create a VAV no reheat terminal and attach the zone/terminal pair to the air loop
        diffuser = openstudio_model.AirTerminalSingleDuctVAVNoReheat(
            model, model.alwaysOnDiscreteSchedule())
        diffuser.setName('{} Diffuser'.format(loop_name))
        air_loop.multiAddBranchForZone(zone, diffuser.to_HVACComponent().get())
        air_loops.append(air_loop)

    return air_loops


def model_add_minisplit_hp(
        model, thermal_zones,
        cooling_type='Two Speed DX AC', heating_type='Single Speed DX',
        hvac_op_sch=None):
    """Creates a minisplit heatpump system for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to connect to this system.
        cooling_type: [String] valid choices are Two Speed DX AC, Single
            Speed DX AC, Single Speed Heat Pump.
        heating_type: [String] valid choices are Single Speed DX.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in
            which case will be defaulted to always on.
    """
    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # default design temperatures across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    # adjusted temperatures for minisplit
    dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
    dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
    dsgn_temps['htg_dsgn_sup_air_temp_f'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_f']
    dsgn_temps['htg_dsgn_sup_air_temp_c'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_c']

    minisplit_hps = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        air_loop = openstudio_model.AirLoopHVAC(model)
        air_loop.setName('{} Minisplit Heat Pump'.format(zone_name))
        loop_name = air_loop.nameString()

        # default design settings used across all air loops
        sizing_system = adjust_sizing_system(
            air_loop, dsgn_temps, sizing_option='NonCoincident')
        sizing_system.setAllOutdoorAirinCooling(False)
        sizing_system.setAllOutdoorAirinHeating(False)

        # create heating coil
        if heating_type == 'Single Speed DX':
            htg_coil = create_coil_heating_dx_single_speed(
                model, name='{} Heating Coil'.format(loop_name),
                type='Residential Minisplit HP')
            o_dbt_fco = TEMPERATURE.to_unit([-30.0], 'C', 'F')[0]
            htg_coil.setMinimumOutdoorDryBulbTemperatureforCompressorOperation(o_dbt_fco)
            o_dbt_fdo = TEMPERATURE.to_unit([-30.0], 'C', 'F')[0]
            htg_coil.setMaximumOutdoorDryBulbTemperatureforDefrostOperation(o_dbt_fdo)
            htg_coil.setCrankcaseHeaterCapacity(0)
            htg_coil.setDefrostStrategy('ReverseCycle')
            htg_coil.setDefrostControl('OnDemand')
            htg_coil.resetDefrostTimePeriodFraction()
        else:
            msg = 'No heating coil type selected for minisplit HP: {}.'.format(zone_name)
            print(msg)
            htg_coil = None

        # create backup heating coil
        supplemental_htg_coil = create_coil_heating_electric(
            model, name='{} Electric Backup Htg Coil'.format(loop_name))

        # create cooling coil
        if cooling_type == 'Two Speed DX AC':
            clg_coil = create_coil_cooling_dx_two_speed(
                model, name='{} 2spd DX AC Clg Coil'.format(loop_name),
                type='Residential Minisplit HP')
        elif cooling_type == 'Single Speed DX AC':
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} 1spd DX AC Clg Coil'.format(loop_name), type='Split AC')
        elif cooling_type == 'Single Speed Heat Pump':
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} 1spd DX HP Clg Coil'.format(loop_name), type='Heat Pump')
        else:
            msg = 'No cooling coil type selected for minisplit HP: {}.'.format(zone_name)
            print(msg)
            clg_coil = None

        # create fan
        fan = create_fan_by_name(
            model, 'Minisplit_HP_Fan', fan_name='{} Fan'.format(loop_name),
            end_use_subcategory='Minisplit HP Fans')
        fan.setAvailabilitySchedule(hvac_op_sch)

        # create unitary system (holds the coils and fan)
        unitary = openstudio_model.AirLoopHVACUnitarySystem(model)
        unitary.setName('{} Unitary System'.format(loop_name))
        unitary.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
        max_sat = TEMPERATURE.to_unit([200.0], 'C', 'F')[0]
        unitary.setMaximumSupplyAirTemperature(max_sat)
        max_dbt_sho = TEMPERATURE.to_unit([40.0], 'C', 'F')[0]
        unitary.setMaximumOutdoorDryBulbTemperatureforSupplementalHeaterOperation(max_dbt_sho)
        unitary.setControllingZoneorThermostatLocation(zone)
        unitary.addToNode(air_loop.supplyInletNode())
        unitary.setSupplyAirFlowRateWhenNoCoolingorHeatingisRequired(0.0)

        # attach the coils and fan
        if htg_coil is not None:
            unitary.setHeatingCoil(htg_coil)
        if clg_coil is not None:
            unitary.setCoolingCoil(clg_coil)
        if supplemental_htg_coil is not None:
            unitary.setSupplementalHeatingCoil(supplemental_htg_coil)
        unitary.setSupplyFan(fan)
        unitary.setFanPlacement('BlowThrough')
        unitary.setSupplyAirFanOperatingModeSchedule(model.alwaysOffDiscreteSchedule())

        # create a diffuser
        if model.version() < openstudio.VersionString('2.7.0'):
            diffuser = openstudio_model.AirTerminalSingleDuctUncontrolled(
                model, model.alwaysOnDiscreteSchedule())
        else:
            diffuser = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                model, model.alwaysOnDiscreteSchedule())
        diffuser.setName('{} Direct Air'.format(zone_name))
        air_loop.multiAddBranchForZone(zone, diffuser.to_HVACComponent().get())

        minisplit_hps.append(air_loop)

    return minisplit_hps


def model_add_ptac(
        model, thermal_zones, cooling_type='Two Speed DX AC', heating_type='Gas',
        hot_water_loop=None, fan_type='Cycling', ventilation=True):
    """Creates a PTAC system for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        cooling_type: [String] valid choices are Two Speed DX AC, Single Speed DX AC.
        heating_type: [String] valid choices are NaturalGas, Electricity,
            Water, nil (no heat).
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to connect
            heating coil to. Set to nil for heating types besides water.
        fan_type: [String] valid choices are ConstantVolume, Cycling.
        ventilation: [Boolean] If True, ventilation will be supplied through
            the unit. If False, no ventilation will be supplied through the unit,
            with the expectation that it will be provided by a DOAS or separate system.
    """
    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()
    if hot_water_loop is not None:
        hw_temp_c = hot_water_loop.sizingPlant().designLoopExitTemperature()
        hw_delta_t_k = hot_water_loop.sizingPlant().loopDesignTemperatureDifference()

    # adjusted zone design temperatures for ptac
    dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
    dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
    dsgn_temps['zn_clg_dsgn_sup_air_temp_f'] = 57.0
    dsgn_temps['zn_clg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_clg_dsgn_sup_air_temp_f']], 'C', 'F')[0]

    # make a PTAC for each zone
    ptacs = []
    for zone in thermal_zones:
        zone_name = zone.nameString()

        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneCoolingDesignSupplyAirHumidityRatio(0.008)
        sizing_zone.setZoneHeatingDesignSupplyAirHumidityRatio(0.008)

        # add fan
        if fan_type == 'ConstantVolume':
            fan = create_fan_by_name(model, 'PTAC_CAV_Fan',
                                     fan_name='{} PTAC Fan'.format(zone_name))
        elif fan_type == 'Cycling':
            fan = create_fan_by_name(model, 'PTAC_Cycling_Fan',
                                     fan_name='{} PTAC Fan'.format(zone_name))
        else:
            raise ValueError('ptac_fan_type "{}" is not recognized.'.format(fan_type))
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        # add heating coil
        if heating_type in ('NaturalGas', 'Gas'):
            htg_coil = create_coil_heating_gas(
                model, name='{} PTAC Gas Htg Coil'.format(zone_name))
        elif heating_type in ('Electricity', 'Electric'):
            htg_coil = create_coil_heating_electric(
                model, name='{} PTAC Electric Htg Coil'.format(zone_name))
        elif heating_type is None:
            htg_coil = create_coil_heating_electric(
                model, name='{} PTAC No Heat'.format(zone_name),
                schedule=model.alwaysOffDiscreteSchedule(), nominal_capacity=0)
        elif heating_type == 'Water':
            if hot_water_loop is None:
                raise ValueError('No hot water plant loop supplied for PTAC water coil.')
            htg_coil = create_coil_heating_water(
                model, hot_water_loop,
                name='{} Water Htg Coil'.format(hot_water_loop.nameString()),
                rated_inlet_water_temperature=hw_temp_c,
                rated_outlet_water_temperature=hw_temp_c - hw_delta_t_k)
        else:
            msg = 'ptac_heating_type of {} is not recognized.'.format(heating_type)
            raise ValueError(msg)

        # add cooling coil
        if cooling_type == 'Two Speed DX AC':
            clg_coil = create_coil_cooling_dx_two_speed(
                model, name='{} PTAC 2spd DX AC Clg Coil'.format(zone_name))
        elif cooling_type == 'Single Speed DX AC':
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} PTAC 1spd DX AC Clg Coil'.format(zone_name), type='PTAC')
        else:
            msg = 'ptac_cooling_type of "{}" is not recognized.'.format(cooling_type)
            raise ValueError(msg)

        # wrap coils in a PTAC system
        ptac_system = openstudio_model.ZoneHVACPackagedTerminalAirConditioner(
            model, model.alwaysOnDiscreteSchedule(), fan, htg_coil, clg_coil)
        ptac_system.setName('{} PTAC'.format(zone_name))
        ptac_system.setFanPlacement('DrawThrough')
        if fan_type == 'ConstantVolume':
            ptac_system.setSupplyAirFanOperatingModeSchedule(
                model.alwaysOnDiscreteSchedule())
        else:
            ptac_system.setSupplyAirFanOperatingModeSchedule(
                model.alwaysOffDiscreteSchedule())
        if not ventilation:
            ptac_system.setOutdoorAirFlowRateDuringCoolingOperation(0.0)
            ptac_system.setOutdoorAirFlowRateDuringHeatingOperation(0.0)
            ptac_system.setOutdoorAirFlowRateWhenNoCoolingorHeatingisNeeded(0.0)
        ptac_system.addToThermalZone(zone)
        ptacs.append(ptac_system)

    return ptacs


def model_add_pthp(model, thermal_zones, fan_type='Cycling', ventilation=True):
    """Creates a PTHP system for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        fan_type: [String] valid choices are ConstantVolume, Cycling.
        ventilation: [Boolean] If true, ventilation will be supplied through the unit.
            If False, no ventilation will be supplied through the unit, with the
            expectation that it will be provided by a DOAS or separate system.
    """
    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    # adjusted zone design temperatures for pthp
    dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
    dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
    dsgn_temps['zn_clg_dsgn_sup_air_temp_f'] = 57.0
    dsgn_temps['zn_clg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_clg_dsgn_sup_air_temp_f']], 'C', 'F')[0]

    # make a PTHP for each zone
    pthps = []
    for zone in thermal_zones:
        zone_name = zone.nameString()

        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneCoolingDesignSupplyAirHumidityRatio(0.008)
        sizing_zone.setZoneHeatingDesignSupplyAirHumidityRatio(0.008)

        # add fan
        if fan_type == 'ConstantVolume':
            fan = create_fan_by_name(model, 'PTAC_CAV_Fan',
                                     fan_name='{} PTHP Fan'.format(zone_name))
        elif fan_type == 'Cycling':
            fan = create_fan_by_name(model, 'PTAC_Cycling_Fan',
                                     fan_name='{} PTHP Fan'.format(zone_name))
        else:
            raise ValueError('PTHP fan_type of {} is not recognized.'.format(fan_type))
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        # add heating coil
        htg_coil = create_coil_heating_dx_single_speed(
            model, name='{} PTHP Htg Coil'.format(zone_name))
        # add cooling coil
        clg_coil = create_coil_cooling_dx_single_speed(
            model, name='{} PTHP Clg Coil'.format(zone_name), type='Heat Pump')
        # supplemental heating coil
        supplemental_htg_coil = create_coil_heating_electric(
            model, name='{} PTHP Supplemental Htg Coil'.format(zone_name))
        # wrap coils in a PTHP system
        pthp_system = openstudio_model.ZoneHVACPackagedTerminalHeatPump(
            model, model.alwaysOnDiscreteSchedule(), fan, htg_coil, clg_coil,
            supplemental_htg_coil)
        pthp_system.setName('{} PTHP'.format(zone_name))
        pthp_system.setFanPlacement('DrawThrough')
        if fan_type == 'ConstantVolume':
            pthp_system.setSupplyAirFanOperatingModeSchedule(
                model.alwaysOnDiscreteSchedule())
        else:
            pthp_system.setSupplyAirFanOperatingModeSchedule(
                model.alwaysOffDiscreteSchedule())
        if not ventilation:
            pthp_system.setOutdoorAirFlowRateDuringCoolingOperation(0.0)
            pthp_system.setOutdoorAirFlowRateDuringHeatingOperation(0.0)
            pthp_system.setOutdoorAirFlowRateWhenNoCoolingorHeatingisNeeded(0.0)
        pthp_system.addToThermalZone(zone)
        pthps.append(pthp_system)

    return pthps


def model_add_unitheater(
        model, thermal_zones, hvac_op_sch=None, fan_control_type='ConstantVolume',
        fan_pressure_rise=0.2, heating_type=None, hot_water_loop=None,
        rated_inlet_water_temperature=180.0, rated_outlet_water_temperature=160.0,
        rated_inlet_air_temperature=60.0, rated_outlet_air_temperature=104.0):
    """Creates a unit heater for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        hvac_op_sch: [String] name of the HVAC operation schedule or None in which
            case will be defaulted to always on.
        fan_control_type: [String] valid choices are OnOff, ConstantVolume,
            VariableVolume.
        fan_pressure_rise: [Double] fan pressure rise, inH2O.
        heating_type: [String] valid choices are NaturalGas, Gas, Electricity,
            Electric, DistrictHeating, DistrictHeatingWater, DistrictHeatingSteam.
        hot_water_loop: [OpenStudio::Model::PlantLoop] hot water loop to connect
            to the heating coil.
        rated_inlet_water_temperature: [Double] rated inlet water temperature in
            degrees Fahrenheit, default is 180F.
        rated_outlet_water_temperature: [Double] rated outlet water temperature
            in degrees Fahrenheit, default is 160F.
        rated_inlet_air_temperature: [Double] rated inlet air temperature in
            degrees Fahrenheit, default is 60F.
        rated_outlet_air_temperature: [Double] rated outlet air temperature in
            degrees Fahrenheit, default is 100F.
    """
    # hvac operation schedule
    hvac_op_sch = model_add_schedule(model, hvac_op_sch)

    # set defaults if nil
    fan_control_type = 'ConstantVolume' if fan_control_type is None else fan_control_type
    fan_pressure_rise = 0.2 if fan_pressure_rise is None else fan_pressure_rise

    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    # adjusted zone design heating temperature for unit heater
    dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
    dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]

    # make a unit heater for each zone
    unit_heaters = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

        # add fan
        fan = create_fan_by_name(
            model, 'Unit_Heater_Fan', fan_name='{} UnitHeater Fan'.format(zone_name),
            pressure_rise=fan_pressure_rise)
        fan.setAvailabilitySchedule(hvac_op_sch)

        # add heating coil
        if heating_type == 'NaturalGas' or heating_type == 'Gas':
            htg_coil = create_coil_heating_gas(
                model, name='{} UnitHeater Gas Htg Coil'.format(zone_name),
                schedule=hvac_op_sch)
        elif heating_type == 'Electricity' or heating_type == 'Electric':
            htg_coil = create_coil_heating_electric(
                model, name='{} UnitHeater Electric Htg Coil'.format(zone_name),
                schedule=hvac_op_sch)
        elif 'DistrictHeating' in heating_type and hot_water_loop is not None:
            # control temperature for hot water loop
            if rated_inlet_water_temperature is None:
                rated_inlet_water_temperature_c = \
                    TEMPERATURE.to_unit([180.0], 'C', 'F')[0]
            else:
                rated_inlet_water_temperature_c = \
                    TEMPERATURE.to_unit([rated_inlet_water_temperature], 'C', 'F')[0]

            if rated_outlet_water_temperature is None:
                rated_outlet_water_temperature_c = \
                    TEMPERATURE.to_unit([160.0], 'C', 'F')[0]
            else:
                rated_outlet_water_temperature_c = \
                    TEMPERATURE.to_unit([rated_outlet_water_temperature], 'C', 'F')[0]

            if rated_inlet_air_temperature is None:
                rated_inlet_air_temperature_c = \
                    TEMPERATURE.to_unit([60.0], 'C', 'F')[0]
            else:
                rated_inlet_air_temperature_c = \
                    TEMPERATURE.to_unit([rated_inlet_air_temperature], 'C', 'F')[0]

            if rated_outlet_air_temperature is None:
                rated_outlet_air_temperature_c = \
                    TEMPERATURE.to_unit([104.0], 'C', 'F')[0]
            else:
                rated_outlet_air_temperature_c = \
                    TEMPERATURE.to_unit([rated_outlet_air_temperature], 'C', 'F')[0]

            htg_coil = create_coil_heating_water(
                model, hot_water_loop,
                name='{} UnitHeater Water Htg Coil'.format(zone_name),
                rated_inlet_water_temperature=rated_inlet_water_temperature_c,
                rated_outlet_water_temperature=rated_outlet_water_temperature_c,
                rated_inlet_air_temperature=rated_inlet_air_temperature_c,
                rated_outlet_air_temperature=rated_outlet_air_temperature_c)
        else:
            raise ValueError('No heating type was found when adding unit heater; '
                             'no unit heater will be created.')

        # create unit heater
        unit_heater = openstudio_model.ZoneHVACUnitHeater(
            model, hvac_op_sch, fan, htg_coil)
        unit_heater.setName('{} Unit Heater'.format(zone_name))
        unit_heater.setFanControlType(fan_control_type)
        unit_heater.addToThermalZone(zone)
        unit_heaters.append(unit_heater)

    return unit_heaters


def model_add_evap_cooler(model, thermal_zones):
    """Creates an evaporative cooler for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
    """
    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    # adjusted design temperatures for evap cooler
    dsgn_temps['clg_dsgn_sup_air_temp_f'] = 70.0
    dsgn_temps['clg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['clg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
    dsgn_temps['max_clg_dsgn_sup_air_temp_f'] = 78.0
    dsgn_temps['max_clg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['max_clg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
    dsgn_temps['approach_r'] = 3.0  # wetbulb approach temperature
    dsgn_temps['approach_k'] = TEMP_DELTA.to_unit([dsgn_temps['approach_r']], 'dC', 'dF')[0]

    # EMS programs
    programs = []

    # Make an evap cooler for each zone
    evap_coolers = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        zone_name_clean = zone_name.replace(':', '')

        # Air loop
        air_loop = openstudio_model.AirLoopHVAC(model)
        air_loop.setName('{} Evaporative Cooler'.format(zone_name_clean))
        loop_name = air_loop.nameString()

        # default design settings used across all air loops
        adjust_sizing_system(air_loop, dsgn_temps)

        # air handler controls
        # setpoint follows OAT WetBulb
        evap_stpt_manager = \
            openstudio_model.SetpointManagerFollowOutdoorAirTemperature(model)
        evap_stpt_manager.setName('{} F above OATwb'.format(dsgn_temps['approach_r']))
        evap_stpt_manager.setReferenceTemperatureType('OutdoorAirWetBulb')
        evap_stpt_manager.setMaximumSetpointTemperature(
            dsgn_temps['max_clg_dsgn_sup_air_temp_c'])
        evap_stpt_manager.setMinimumSetpointTemperature(
            dsgn_temps['clg_dsgn_sup_air_temp_c'])
        evap_stpt_manager.setOffsetTemperatureDifference(dsgn_temps['approach_k'])
        evap_stpt_manager.addToNode(air_loop.supplyOutletNode())

        # Schedule to control the airloop availability
        air_loop_avail_sch = openstudio_model.ScheduleConstant(model)
        air_loop_avail_sch.setName('{} Availability Sch'.format(loop_name))
        air_loop_avail_sch.setValue(1)
        air_loop.setAvailabilitySchedule(air_loop_avail_sch)

        # EMS to turn on Evap Cooler if there is a cooling load in the target zone.
        # Without this EMS, the airloop runs 24/7-365 even when there is no load in the zone.

        # Create a sensor to read the zone load
        ems_zone_name = ems_friendly_name(zone_name_clean)
        zn_load_sensor = openstudio_model.EnergyManagementSystemSensor(
            model, 'Zone Predicted Sensible Load to Cooling Setpoint Heat Transfer Rate')
        zn_load_sensor.setName('{} Clg Load Sensor'.format(ems_zone_name))
        zn_load_sensor.setKeyName(str(zone.handle()))

        # Create an actuator to set the airloop availability
        ems_loop_name = ems_friendly_name(air_loop.name)
        air_loop_avail_actuator = openstudio_model.EnergyManagementSystemActuator(
            air_loop_avail_sch, 'Schedule:Constant', 'Schedule Value')
        air_loop_avail_actuator.setName('{} Availability Actuator'.format(ems_loop_name))

        # Create a program to turn on Evap Cooler if
        # there is a cooling load in the target zone.
        # Load < 0.0 is a cooling load.
        avail_program = openstudio_model.EnergyManagementSystemProgram(model)
        avail_program.setName('{} Availability Control'.format(ems_loop_name))
        avail_program_body = \
            'IF {zn_load_sensor_handle} < 0.0\n' \
            'SET {air_loop_avail_actuator_handle} = 1\n' \
            'ELSE\n' \
            'SET {air_loop_avail_actuator_handle} = 0\n' \
            'ENDIF'.format(
                zn_load_sensor_handle=zn_load_sensor.handle(),
                air_loop_avail_actuator_handle=air_loop_avail_actuator.handle()
            )
        avail_program.setBody(avail_program_body)

        programs.append(avail_program)

        # Direct Evap Cooler
        # @todo better assumptions for fan pressure rise
        evap = openstudio_model.EvaporativeCoolerDirectResearchSpecial(
            model, model.alwaysOnDiscreteSchedule())
        evap.setName('{} Evap Media'.format(zone_name))
        # assume 90% design effectiveness from
        # https://basc.pnnl.gov/resource-guides/evaporative-cooling-systems#edit-group-description
        evap.setCoolerDesignEffectiveness(0.90)
        evap.autosizePrimaryAirDesignFlowRate()
        evap.autosizeRecirculatingWaterPumpPowerConsumption()
        # use suggested E+ default values of 90.0 W-s/m^3 for pump sizing factor
        # and 3.0 for blowdown concentration
        evap.setWaterPumpPowerSizingFactor(90.0)
        evap.setBlowdownConcentrationRatio(3.0)
        evap.addToNode(air_loop.supplyInletNode())

        # Fan (cycling), must be inside unitary system to cycle on airloop
        fan = create_fan_by_name(
            model, 'Evap_Cooler_Supply_Fan',
            fan_name='{} Evap Cooler Supply Fan'.format(zone_name))
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        # Dummy zero-capacity cooling coil
        clg_coil = create_coil_cooling_dx_single_speed(
            model, name='Dummy Always Off DX Coil',
            schedule=model.alwaysOffDiscreteSchedule())
        unitary_system = openstudio_model.AirLoopHVACUnitarySystem(model)
        unitary_system.setName('{} Evap Cooler Cycling Fan'.format(zone_name))
        unitary_system.setSupplyFan(fan)
        unitary_system.setCoolingCoil(clg_coil)
        unitary_system.setControllingZoneorThermostatLocation(zone)
        unitary_system.setMaximumSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        unitary_system.setFanPlacement('BlowThrough')
        if model.version() < openstudio.VersionString('3.7.0'):
            unitary_system.setSupplyAirFlowRateMethodDuringCoolingOperation(
                'SupplyAirFlowRate')
            unitary_system.setSupplyAirFlowRateMethodDuringHeatingOperation(
                'SupplyAirFlowRate')
            unitary_system.setSupplyAirFlowRateMethodWhenNoCoolingorHeatingisRequired(
                'SupplyAirFlowRate')
        else:
            unitary_system.autosizeSupplyAirFlowRateDuringCoolingOperation()
            unitary_system.autosizeSupplyAirFlowRateDuringHeatingOperation()
            unitary_system.autosizeSupplyAirFlowRateWhenNoCoolingorHeatingisRequired()
        unitary_system.setSupplyAirFanOperatingModeSchedule(
            model.alwaysOffDiscreteSchedule())
        unitary_system.addToNode(air_loop.supplyInletNode())

        # Outdoor air intake system
        oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
        oa_intake_controller.setName('{} OA Controller'.format(loop_name))
        oa_intake_controller.setMinimumLimitType('FixedMinimum')
        oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
        oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
        oa_intake_controller.setMinimumFractionofOutdoorAirSchedule(
            model.alwaysOnDiscreteSchedule())
        controller_mv = oa_intake_controller.controllerMechanicalVentilation()
        controller_mv.setName('{} Vent Controller'.format(loop_name))
        controller_mv.setSystemOutdoorAirMethod('ZoneSum')

        oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(
            model, oa_intake_controller)
        oa_intake.setName('{} OA System'.format(loop_name))
        oa_intake.addToNode(air_loop.supplyInletNode())

        # make an air terminal for the zone
        if model.version() < openstudio.VersionString('2.7.0'):
            air_terminal = openstudio_model.AirTerminalSingleDuctUncontrolled(
                model, model.alwaysOnDiscreteSchedule())
        else:
            air_terminal = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                model, model.alwaysOnDiscreteSchedule())
        air_terminal.setName('{} Air Terminal'.format(zone_name))

        # attach new terminal to the zone and to the airloop
        air_loop.multiAddBranchForZone(zone, air_terminal.to_HVACComponent().get())

        sizing_zone = zone.sizingZone()
        sizing_zone.setCoolingDesignAirFlowMethod('DesignDay')
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])

        evap_coolers.append(air_loop)

    # Create a programcallingmanager
    avail_pcm = openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    avail_pcm.setName('EvapCoolerAvailabilityProgramCallingManager')
    avail_pcm.setCallingPoint('AfterPredictorAfterHVACManagers')
    for program in programs:
        avail_pcm.addProgram(program)

    return evap_coolers


def model_add_baseboard(
        model, thermal_zones, hot_water_loop=None):
    """Adds hydronic or electric baseboard heating to each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            add baseboards to.
        hot_water_loop: [OpenStudio::Model::PlantLoop] The hot water loop that
            serves the baseboards. If None, baseboards are electric.
    """
    # Make a baseboard heater for each zone
    baseboards = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        if hot_water_loop is None:
            baseboard = openstudio_model.ZoneHVACBaseboardConvectiveElectric(model)
            baseboard.setName('{} Electric Baseboard'.format(zone_name))
            baseboard.addToThermalZone(zone)
            baseboards.append(baseboard)
        else:
            htg_coil = openstudio_model.CoilHeatingWaterBaseboard(model)
            htg_coil.setName('{} Hydronic Baseboard Coil'.format(zone_name))
            hot_water_loop.addDemandBranchForComponent(htg_coil)
            baseboard = openstudio_model.ZoneHVACBaseboardConvectiveWater(
                model, model.alwaysOnDiscreteSchedule(), htg_coil)
            baseboard.setName('{} Hydronic Baseboard'.format(zone_name))
            baseboard.addToThermalZone(zone)
            baseboards.append(baseboard)
    return baseboards


def model_add_vrf(model, thermal_zones, ventilation=False):
    """Adds Variable Refrigerant Flow system and terminal units for each zone

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to add fan coil units.
        ventilation: [Boolean] If True, ventilation will be supplied through the unit.
            If False, no ventilation will be supplied through the unit, with the
            expectation that it will be provided by a DOAS or separate system.
    """
    # create vrf outdoor unit
    master_zone = thermal_zones[0]
    vrf_outdoor_unit = create_air_conditioner_variable_refrigerant_flow(
        model, name='{} Zone VRF System'.format(len(thermal_zones)),
        master_zone=master_zone)

    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    for zone in thermal_zones:
        zone_name = zone.nameString()

        # zone sizing
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

        # add vrf terminal unit
        vrf_terminal_unit = \
            openstudio_model.ZoneHVACTerminalUnitVariableRefrigerantFlow(model)
        vrf_terminal_unit.setName('{} VRF Terminal Unit'.format(zone_name))
        vrf_terminal_unit.addToThermalZone(zone)
        vrf_terminal_unit.setTerminalUnitAvailabilityschedule(
            model.alwaysOnDiscreteSchedule())

        if ventilation is not None:
            vrf_terminal_unit.setOutdoorAirFlowRateDuringCoolingOperation(0.0)
            vrf_terminal_unit.setOutdoorAirFlowRateDuringHeatingOperation(0.0)
            vrf_terminal_unit.setOutdoorAirFlowRateWhenNoCoolingorHeatingisNeeded(0.0)

        # set fan variables
        # always off denotes cycling fan
        vrf_terminal_unit.setSupplyAirFanOperatingModeSchedule(
            model.alwaysOffDiscreteSchedule())
        vrf_fan = vrf_terminal_unit.supplyAirFan().to_FanOnOff()
        if vrf_fan.is_initialized():
            vrf_fan = vrf_fan.get()
            vrf_fan.setPressureRise(300.0)
            vrf_fan.setMotorEfficiency(0.8)
            vrf_fan.setFanEfficiency(0.6)
            vrf_fan.setName('{} VRF Unit Cycling Fan'.format(zone_name))

        # add to main condensing unit
        vrf_outdoor_unit.addTerminal(vrf_terminal_unit)

    return vrf_outdoor_unit


def model_add_four_pipe_fan_coil(
        model, thermal_zones, chilled_water_loop, hot_water_loop=None,
        ventilation=False, capacity_control_method='CyclingFan'):
    """Adds four pipe fan coil units to each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            add fan coil units.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] the chilled water loop
            that serves the fan coils.
        hot_water_loop: [OpenStudio::Model::PlantLoop] the hot water loop that
            serves the fan coils. If None, a zero-capacity, electric heating
            coil set to Always-Off will be included in the unit.
        ventilation: [Boolean] If true, ventilation will be supplied through
            the unit. If false, no ventilation will be supplied through the unit,
            with the expectation that it will be provided by a DOAS or separate system.
        capacity_control_method: [String] Capacity control method for the fan coil.
            Options are ConstantFanVariableFlow, CyclingFan, VariableFanVariableFlow,
            and VariableFanConstantFlow.  If VariableFan, the fan will be VariableVolume.
    """
    # default design temperatures used across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    # make a fan coil unit for each zone
    fcus = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        sizing_zone = zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(
            dsgn_temps['zn_clg_dsgn_sup_air_temp_c'])
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(
            dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])

        if chilled_water_loop:
            fcu_clg_coil = create_coil_cooling_water(
                model, chilled_water_loop, name='{} FCU Cooling Coil'.format(zone_name))
        else:
            raise ValueError('Fan coil units require a chilled water loop, '
                             'but none was provided.')

        if hot_water_loop:
            fcu_htg_coil = create_coil_heating_water(
                model, hot_water_loop, name='{} FCU Heating Coil'.format(zone_name),
                rated_outlet_air_temperature=dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        else:  # Zero-capacity, always-off electric heating coil
            fcu_htg_coil = create_coil_heating_electric(
                model, name='{} No Heat'.format(zone_name),
                schedule=model.alwaysOffDiscreteSchedule(), nominal_capacity=0.0)

        if capacity_control_method in ('VariableFanVariableFlow', 'VariableFanConstantFlow'):
            fcu_fan = create_fan_by_name(
                model, 'Fan_Coil_VarSpeed_Fan',
                fan_name='{} Fan Coil Variable Fan'.format(zone_name),
                end_use_subcategory='FCU Fans')
        else:
            fcu_fan = create_fan_by_name(
                model, 'Fan_Coil_Fan', fan_name='{} Fan Coil fan'.format(zone_name),
                end_use_subcategory='FCU Fans')
        fcu_fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
        fcu_fan.autosizeMaximumFlowRate()

        fcu = openstudio_model.ZoneHVACFourPipeFanCoil(
            model, model.alwaysOnDiscreteSchedule(), fcu_fan, fcu_clg_coil, fcu_htg_coil)
        fcu.setName('{} FCU'.format(zone_name))
        fcu.setCapacityControlMethod(capacity_control_method)
        fcu.autosizeMaximumSupplyAirFlowRate()
        if not ventilation:
            fcu.setMaximumOutdoorAirFlowRate(0.0)
        fcu.addToThermalZone(zone)
        fcus.append(fcu)

    return fcus


def model_add_high_temp_radiant(
        model, thermal_zones, heating_type='NaturalGas',
        combustion_efficiency=0.8, control_type='MeanAirTemperature'):
    """Creates a high temp radiant heater for each zone and adds it to the model.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            connect to this system.
        heating_type: [String] valid choices are NaturalGas, Electric.
        combustion_efficiency: [Double] combustion efficiency as decimal.
        control_type: [String] control type.
    """
    heating_type == 'NaturalGas' if heating_type is None else heating_type
    # make a high temp radiant heater for each zone
    radiant_heaters = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        high_temp_radiant = openstudio_model.ZoneHVACHighTemperatureRadiant(model)
        high_temp_radiant.setName('{} High Temp Radiant'.format(zone_name))

        if heating_type == 'NaturalGas' or heating_type == 'Gas':
            high_temp_radiant.setFuelType('NaturalGas')
        else:
            high_temp_radiant.setFuelType(heating_type)

        if combustion_efficiency is None:
            if heating_type == 'NaturalGas' or heating_type == 'Gas':
                high_temp_radiant.setCombustionEfficiency(0.8)
            elif heating_type == 'Electric':
                high_temp_radiant.setCombustionEfficiency(1.0)
        else:
            high_temp_radiant.setCombustionEfficiency(combustion_efficiency)

        # set heating setpoint schedule
        htg_sch = None
        tstat = zone.thermostatSetpointDualSetpoint()
        if tstat.is_initialized():
            tstat = tstat.get()
            if tstat.heatingSetpointTemperatureSchedule().is_initialized():
                htg_sch = tstat.heatingSetpointTemperatureSchedule().get()
        if htg_sch is None:
            msg = 'For {}: Cannot find a heating setpoint schedule for this zone, ' \
                'cannot apply high temp radiant system.'.format(zone_name)
            raise ValueError(msg)

        # set defaults
        high_temp_radiant.setHeatingSetpointTemperatureSchedule(htg_sch)
        high_temp_radiant.setTemperatureControlType(control_type)
        high_temp_radiant.setFractionofInputConvertedtoRadiantEnergy(0.8)
        high_temp_radiant.setHeatingThrottlingRange(2)
        high_temp_radiant.addToThermalZone(zone)
        radiant_heaters.append(high_temp_radiant)

    return radiant_heaters


def model_add_low_temp_radiant(
        model, thermal_zones, hot_water_loop, chilled_water_loop,
        plant_supply_water_temperature_control=False,
        plant_supply_water_temperature_control_strategy='outdoor_air',
        hwsp_at_oat_low=120.0, hw_oat_low=55.0, hwsp_at_oat_high=80.0, hw_oat_high=70.0,
        chwsp_at_oat_low=70.0, chw_oat_low=65.0, chwsp_at_oat_high=55.0, chw_oat_high=75.0,
        radiant_type='floor', radiant_temperature_control_type='SurfaceFaceTemperature',
        radiant_setpoint_control_type='ZeroFlowPower',
        include_carpet=True, carpet_thickness_in=0.25,
        control_strategy='proportional_control',
        use_zone_occupancy_for_control=True, occupied_percentage_threshold=0.10,
        model_occ_hr_start=6.0, model_occ_hr_end=18.0,
        proportional_gain=0.3, switch_over_time=24.0,
        slab_sp_at_oat_low=73, slab_oat_low=65,
        slab_sp_at_oat_high=68, slab_oat_high=80,
        radiant_availability_type='precool', radiant_lockout=False,
        radiant_lockout_start_time=12.0, radiant_lockout_end_time=20.0):
    """Adds low temperature radiant loop systems to each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            add radiant loops.
        hot_water_loop: [OpenStudio::Model::PlantLoop] the hot water loop that
            serves the radiant loop.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] the chilled water loop
            that serves the radiant loop.
        plant_supply_water_temperature_control: [Bool] Set to True if the plant
            supply water temperature is to be controlled else it is held constant,
            default to false.
        plant_supply_water_temperature_control_strategy: [String] Method to determine
            how to control the plant's supply water temperature. Options include.
            outdoor_air - Set the supply water temperature based on the outdoor
                air temperature.
            zone_demand - Set the supply water temperature based on the preponderance
                of zone demand.
        hwsp_at_oat_low: [Double] hot water plant supply water temperature setpoint,
            in F, at the outdoor low temperature.
        hw_oat_low: [Double] hot water plant supply water temperature setpoint,
            in F, at the outdoor high temperature.
        hw_oat_high: [Double] outdoor drybulb air temperature, in F, for high
            setpoint for hot water plant.
        chwsp_at_oat_low: [Double] chilled water plant supply water temperature
            setpoint, in F, at the outdoor low temperature.
        chw_oat_low: [Double] outdoor drybulb air  temperature, in F, for low
            setpoint for chilled water plant.
        chwsp_at_oat_high: [Double] chilled water plant supply water temperature
            setpoint, in F, at the outdoor high temperature.
        chw_oat_high: [Double] outdoor drybulb air temperature, in F, for high
            setpoint for chilled water plant.
        radiant_type: [String] type of radiant system to create in zone. Valid options
            incllude the following.

            * floor
            * ceiling
            * ceilingmetalpanel
            * floorwithhardwood

        radiant_temperature_control_type: [String] determines the controlled
            temperature for the radiant system options include the following.

            * MeanAirTemperature
            * MeanRadiantTemperature
            * OperativeTemperature
            * OutdoorDryBulbTemperature
            * OutdoorWetBulbTemperature
            * SurfaceFaceTemperature
            * SurfaceInteriorTemperature

        radiant_setpoint_control_type: [String] determines the response of the
            radiant system at setpoint temperature options include the following.

            * ZeroFlowPower
            * HalfFlowPower

        include_carpet: [Boolean] boolean to include thin carpet tile over
            radiant slab, default to true.
        carpet_thickness_in: [Double] thickness of carpet in inches.
        control_strategy: [String] name of control strategy. Options include
            the following.

            * proportional_control
            * oa_based_control
            * constant_control
            * none

            If control strategy is proportional_control, the method will apply
            the CBE radiant control sequences detailed in Raftery et al. (2017)
            'A new control strategy for high thermal mass radiant systems'.
            If control strategy is oa_based_control, the method will apply native
            EnergyPlus objects/parameters to vary slab setpoint based on outdoor weather.
            If control strategy is constant_control, the method will apply native
            EnergyPlus objects/parameters to maintain a constant slab setpoint.
            Otherwise no control strategy will be applied and the radiant system
            will assume the EnergyPlus default controls.
        use_zone_occupancy_for_control: [Boolean] Set to true if radiant system is
            to use specific zone occupancy objects for CBE control strategy. If False,
            then it will use values in model_occ_hr_start and model_occ_hr_end
            for all radiant zones. default to True.
        occupied_percentage_threshold: [Double] the minimum fraction (0 to 1) that
            counts as occupied if this parameter is set, the returned ScheduleRuleset
            will be 0 = unoccupied, 1 = occupied. Otherwise the ScheduleRuleset
            will be the weighted fractional occupancy schedule. Only used if
            use_zone_occupancy_for_control is set to true.
        model_occ_hr_start: [Double] (Optional) Only applies if control_strategy
            is proportional_control. Starting hour of building occupancy.
        model_occ_hr_end: [Double] (Optional) Only applies if control_strategy
            is proportional_control. Ending hour of building occupancy.
        proportional_gain: [Double] (Optional) Only applies if control_strategy
            is proportional_control. Proportional gain constant (recommended 0.3 or less).
        switch_over_time: [Double] Time limitation for when the system can switch
            between heating and cooling
        slab_sp_at_oat_low: [Double] radiant slab temperature setpoint, in F,
            at the outdoor high temperature.
        slab_oat_low: [Double] outdoor drybulb air temperature, in F, for low
            radiant slab setpoint.
        slab_sp_at_oat_high: [Double] radiant slab temperature setpoint, in F,
            at the outdoor low temperature.
        slab_oat_high: [Double] outdoor drybulb air temperature, in F, for high
            radiant slab setpoint.
        radiant_availability_type: [String] a preset that determines the availability
            of the radiant system. Options include the following.

            * all_day - radiant system is available 24 hours a day
            * precool - primarily operates radiant system during night-time hours
            * afternoon_shutoff - avoids operation during peak grid demand
            * occupancy - operates radiant system during building occupancy hours

        radiant_lockout: [Boolean] True if system contains a radiant lockout.
            If true, it will overwrite radiant_availability_type.
        radiant_lockout_start_time: [double] decimal hour of when radiant lockout starts.
            Only used if radiant_lockout is True.
        radiant_lockout_end_time: [double] decimal hour of when radiant lockout ends.
            Only used if radiant_lockout is True.
    """
    # determine construction insulation thickness by climate zone
    climate_zone_obj = model.getClimateZones().getClimateZone('ASHRAE', 2006)
    if climate_zone_obj.value() == '':
        climate_zone_obj = model.getClimateZones().getClimateZone('ASHRAE', 2013)
    if climate_zone_obj.value() == '':
        climate_zone = None
    else:
        climate_zone = climate_zone_obj.value()

    if climate_zone is None:
        msg = 'Unable to determine climate zone for radiant slab insulation ' \
            'determination. Defaulting to climate zone 5, R-20 insulation, 110F ' \
            'heating design supply water temperature.'
        print(msg)
        cz_mult = 4
        radiant_htg_dsgn_sup_wtr_temp_f = 110
    else:
        if climate_zone in ('0', '1'):
            cz_mult = 2
            radiant_htg_dsgn_sup_wtr_temp_f = 90
        elif climate_zone in ('2', '2A', '2B'):
            cz_mult = 2
            radiant_htg_dsgn_sup_wtr_temp_f = 100
        elif climate_zone in ('3', '3A', '3B', '3C'):
            cz_mult = 3
            radiant_htg_dsgn_sup_wtr_temp_f = 100
        elif climate_zone in ('4', '4A', '4B', '4C'):
            cz_mult = 4
            radiant_htg_dsgn_sup_wtr_temp_f = 100
        elif climate_zone in ('5', '5A', '5B', '5C'):
            cz_mult = 4
            radiant_htg_dsgn_sup_wtr_temp_f = 110
        elif climate_zone in ('6', '6A', '6B'):
            cz_mult = 4
            radiant_htg_dsgn_sup_wtr_temp_f = 120
        elif climate_zone in ('7', '8'):
            cz_mult = 5
            radiant_htg_dsgn_sup_wtr_temp_f = 120
        else:  # default to 4
            cz_mult = 4
            radiant_htg_dsgn_sup_wtr_temp_f = 100

    # create materials
    mat_concrete_3_5in = openstudio_model.StandardOpaqueMaterial(
        model, 'MediumRough', 0.0889, 2.31, 2322, 832)
    mat_concrete_3_5in.setName('Radiant Slab Concrete - 3.5 in.')

    mat_concrete_1_5in = openstudio_model.StandardOpaqueMaterial(
        model, 'MediumRough', 0.0381, 2.31, 2322, 832)
    mat_concrete_1_5in.setName('Radiant Slab Concrete - 1.5 in')

    metal_mat = None
    air_gap_mat = None
    wood_mat = None
    wood_floor_insulation = None
    gypsum_ceiling_mat = None
    if radiant_type == 'ceilingmetalpanel':
        metal_mat = openstudio_model.StandardOpaqueMaterial(
            model, 'MediumSmooth', 0.003175, 30, 7680, 418)
        metal_mat.setName('Radiant Metal Layer - 0.125 in')
        air_gap_mat = openstudio_model.MasslessOpaqueMaterial(model, 'Smooth', 0.004572)
        air_gap_mat.setName('Generic Ceiling Air Gap - R 0.025')
    elif radiant_type == 'floorwithhardwood':
        wood_mat = openstudio_model.StandardOpaqueMaterial(
            model, 'MediumSmooth', 0.01905, 0.15, 608, 1629)
        wood_mat.setName('Radiant Hardwood Flooring - 0.75 in')
        wood_floor_insulation = openstudio_model.StandardOpaqueMaterial(
            model, 'Rough', 0.0508, 0.02, 56.06, 1210)
        wood_floor_insulation.setName('Radiant Subfloor Insulation - 4.0 in')
        gypsum_ceiling_mat = openstudio_model.StandardOpaqueMaterial(
            model, 'Smooth', 0.0127, 0.16, 800, 1089)
        gypsum_ceiling_mat.setName('Gypsum Ceiling for Radiant Hardwood Flooring - 0.5 in')

    mat_refl_roof_membrane = model.getStandardOpaqueMaterialByName(
        'Roof Membrane - Highly Reflective')
    if mat_refl_roof_membrane.is_initialized():
        mat_refl_roof_membrane = mat_refl_roof_membrane.get()
    else:
        mat_refl_roof_membrane = openstudio_model.StandardOpaqueMaterial(
            model, 'VeryRough', 0.0095, 0.16, 1121.29, 1460)
        mat_refl_roof_membrane.setThermalAbsorptance(0.75)
        mat_refl_roof_membrane.setSolarAbsorptance(0.45)
        mat_refl_roof_membrane.setVisibleAbsorptance(0.7)
        mat_refl_roof_membrane.setName('Roof Membrane - Highly Reflective')

    if include_carpet:
        carpet_thickness_m = TEMPERATURE.to_unit([carpet_thickness_in / 12.0], 'm', 'ft')[0]
        conductivity_si = 0.06
        mat_thin_carpet_tile = openstudio_model.StandardOpaqueMaterial(
            model, 'MediumRough', carpet_thickness_m, conductivity_si, 288, 1380)
        mat_thin_carpet_tile.setThermalAbsorptance(0.9)
        mat_thin_carpet_tile.setSolarAbsorptance(0.7)
        mat_thin_carpet_tile.setVisibleAbsorptance(0.8)
        mat_thin_carpet_tile.setName('Radiant Slab Thin Carpet Tile')

    # set exterior slab insulation thickness based on climate zone
    slab_insulation_thickness_m = 0.0254 * cz_mult
    mat_slab_insulation = openstudio_model.StandardOpaqueMaterial(
        model, 'Rough', slab_insulation_thickness_m, 0.02, 56.06, 1210)
    slab_in_name = 'Radiant Ground Slab Insulation - {} in.'.format(cz_mult)
    mat_slab_insulation.setName(slab_in_name)

    ext_insulation_thickness_m = 0.0254 * (cz_mult + 1)
    mat_ext_insulation = openstudio_model.StandardOpaqueMaterial(
        model, 'Rough', ext_insulation_thickness_m, 0.02, 56.06, 1210)
    ext_in_name = 'Radiant Exterior Slab Insulation - {} in.'.format(cz_mult + 1)
    mat_ext_insulation.setName(ext_in_name)

    roof_insulation_thickness_m = 0.0254 * (cz_mult + 1) * 2
    mat_roof_insulation = openstudio_model.StandardOpaqueMaterial(
        model, 'Rough', roof_insulation_thickness_m, 0.02, 56.06, 1210)
    rf_in_name = 'Radiant Exterior Ceiling Insulation - {} in.'.format((cz_mult + 1) * 2)
    mat_roof_insulation.setName(rf_in_name)

    # create radiant internal source constructions
    # create radiant internal source constructions
    radiant_ground_slab_construction = None
    radiant_exterior_slab_construction = None
    radiant_interior_floor_slab_construction = None
    radiant_interior_ceiling_slab_construction = None
    radiant_ceiling_slab_construction = None
    radiant_interior_ceiling_metal_construction = None
    radiant_ceiling_metal_construction = None

    if radiant_type == 'floor':
        layers = [mat_slab_insulation, mat_concrete_3_5in, mat_concrete_1_5in]
        if include_carpet:
            layers.append(mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_ground_slab_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_ground_slab_construction.setLayers(materials)
        radiant_ground_slab_construction.setName('Radiant Ground Slab Construction')
        radiant_ground_slab_construction.setSourcePresentAfterLayerNumber(2)
        radiant_ground_slab_construction.setTemperatureCalculationRequestedAfterLayerNumber(3)
        radiant_ground_slab_construction.setTubeSpacing(0.2286)  # 9 inches

        layers = [mat_ext_insulation, mat_concrete_3_5in, mat_concrete_1_5in]
        if include_carpet:
            layers.append(mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_exterior_slab_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_exterior_slab_construction.setLayers(materials)
        radiant_exterior_slab_construction.setName('Radiant Exterior Slab Construction')
        radiant_exterior_slab_construction.setSourcePresentAfterLayerNumber(2)
        radiant_exterior_slab_construction.setTemperatureCalculationRequestedAfterLayerNumber(3)
        radiant_exterior_slab_construction.setTubeSpacing(0.2286)  # 9 inches

        layers = [mat_concrete_3_5in, mat_concrete_1_5in]
        if include_carpet:
            layers.append(mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_interior_floor_slab_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_interior_floor_slab_construction.setLayers(materials)
        radiant_interior_floor_slab_construction.setName(
            'Radiant Interior Floor Slab Construction')
        radiant_interior_floor_slab_construction.setSourcePresentAfterLayerNumber(1)
        radiant_interior_floor_slab_construction.setTemperatureCalculationRequestedAfterLayerNumber(2)
        radiant_interior_floor_slab_construction.setTubeSpacing(0.2286)  # 9 inches

    elif radiant_type == 'ceiling':
        layers = [mat_concrete_3_5in, mat_concrete_1_5in]
        if include_carpet:
            layers.insert(0, mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_interior_ceiling_slab_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_interior_ceiling_slab_construction.setLayers(materials)
        radiant_interior_ceiling_slab_construction.setName(
            'Radiant Interior Ceiling Slab Construction')
        slab_src_loc = 2 if include_carpet else 1
        radiant_interior_ceiling_slab_construction.setSourcePresentAfterLayerNumber(slab_src_loc)
        radiant_interior_ceiling_slab_construction.setTemperatureCalculationRequestedAfterLayerNumber(
            slab_src_loc + 1)
        radiant_interior_ceiling_slab_construction.setTubeSpacing(0.2286)  # 9 inches

        layers = [mat_refl_roof_membrane, mat_roof_insulation,
                  mat_concrete_3_5in, mat_concrete_1_5in]
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_ceiling_slab_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_ceiling_slab_construction.setLayers(materials)
        radiant_ceiling_slab_construction.setName(
            'Radiant Exterior Ceiling Slab Construction')
        radiant_ceiling_slab_construction.setSourcePresentAfterLayerNumber(3)
        radiant_ceiling_slab_construction.setTemperatureCalculationRequestedAfterLayerNumber(4)
        radiant_ceiling_slab_construction.setTubeSpacing(0.2286)  # 9 inches

    elif radiant_type == 'ceilingmetalpanel':
        layers = [mat_concrete_3_5in, air_gap_mat, metal_mat, metal_mat]
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_interior_ceiling_metal_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_interior_ceiling_metal_construction.setLayers(materials)
        radiant_interior_ceiling_metal_construction.setName(
            'Radiant Interior Ceiling Metal Construction')
        radiant_interior_ceiling_metal_construction.setSourcePresentAfterLayerNumber(3)
        radiant_interior_ceiling_metal_construction.setTemperatureCalculationRequestedAfterLayerNumber(4)
        radiant_interior_ceiling_metal_construction.setTubeSpacing(0.1524)  # 6 inches

        layers = [mat_refl_roof_membrane, mat_roof_insulation, mat_concrete_3_5in,
                  air_gap_mat, metal_mat, metal_mat]
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_ceiling_metal_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_ceiling_metal_construction.setLayers(materials)
        radiant_ceiling_metal_construction.setName('Radiant Ceiling Metal Construction')
        radiant_ceiling_metal_construction.setSourcePresentAfterLayerNumber(5)
        radiant_ceiling_metal_construction.setTemperatureCalculationRequestedAfterLayerNumber(6)
        radiant_ceiling_metal_construction.setTubeSpacing(0.1524)  # 6 inches

    elif radiant_type == 'floorwithhardwood':
        layers = [mat_slab_insulation, mat_concrete_3_5in, wood_mat]
        if include_carpet:
            layers.append(mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_ground_wood_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_ground_wood_construction.setLayers(materials)
        radiant_ground_wood_construction.setName(
            'Radiant Ground Slab Wood Floor Construction')
        radiant_ground_wood_construction.setSourcePresentAfterLayerNumber(2)
        radiant_ground_wood_construction.setTemperatureCalculationRequestedAfterLayerNumber(3)
        radiant_ground_wood_construction.setTubeSpacing(0.2286)  # 9 inches

        layers = [mat_ext_insulation, wood_mat]
        if include_carpet:
            layers.append(mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_exterior_wood_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_exterior_wood_construction.setLayers(materials)
        radiant_exterior_wood_construction.setName(
            'Radiant Exterior Wood Floor Construction')
        radiant_exterior_wood_construction.setSourcePresentAfterLayerNumber(1)
        radiant_exterior_wood_construction.setTemperatureCalculationRequestedAfterLayerNumber(2)
        radiant_exterior_wood_construction.setTubeSpacing(0.2286)  # 9 inches

        layers = [gypsum_ceiling_mat, wood_floor_insulation, wood_mat]
        if include_carpet:
            layers.append(mat_thin_carpet_tile)
        materials = os_create_vector(layers, openstudio_model.MaterialVector())
        radiant_interior_wood_floor_construction = \
            openstudio_model.ConstructionWithInternalSource(model)
        radiant_interior_wood_floor_construction.setLayers(materials)
        radiant_interior_wood_floor_construction.setName(
            'Radiant Interior Wooden Floor Construction')
        radiant_interior_wood_floor_construction.setSourcePresentAfterLayerNumber(2)
        radiant_interior_wood_floor_construction.setTemperatureCalculationRequestedAfterLayerNumber(3)
        radiant_interior_wood_floor_construction.setTubeSpacing(0.2286)  # 9 inches

    # adjust hot and chilled water loop temperatures and set new setpoint schedules
    radiant_htg_dsgn_sup_wtr_temp_delt_r = 10.0
    radiant_htg_dsgn_sup_wtr_temp_c = \
        TEMPERATURE.to_unit([radiant_htg_dsgn_sup_wtr_temp_f], 'C', 'F')[0]
    radiant_htg_dsgn_sup_wtr_temp_delt_k = \
        TEMP_DELTA.to_unit([radiant_htg_dsgn_sup_wtr_temp_delt_r], 'dC', 'dF')[0]
    hot_water_loop.sizingPlant().setDesignLoopExitTemperature(
        radiant_htg_dsgn_sup_wtr_temp_c)
    hot_water_loop.sizingPlant().setLoopDesignTemperatureDifference(
        radiant_htg_dsgn_sup_wtr_temp_delt_k)
    hw_sch_name = '{} Temp - {}F'.format(
        hot_water_loop.nameString(), round(radiant_htg_dsgn_sup_wtr_temp_f))
    hw_temp_sch = create_constant_schedule_ruleset(
        model, radiant_htg_dsgn_sup_wtr_temp_c, name=hw_sch_name,
        schedule_type_limit='Temperature')
    for spm in hot_water_loop.supplyOutletNode().setpointManagers():
        if spm.to_SetpointManagerScheduled().is_initialized():
            spm = spm.to_SetpointManagerScheduled().get()
            spm.setSchedule(hw_temp_sch)

    radiant_clg_dsgn_sup_wtr_temp_f = 55.0
    radiant_clg_dsgn_sup_wtr_temp_delt_r = 5.0
    radiant_clg_dsgn_sup_wtr_temp_c = \
        TEMPERATURE.to_unit([radiant_clg_dsgn_sup_wtr_temp_f], 'C', 'F')[0]
    radiant_clg_dsgn_sup_wtr_temp_delt_k = \
        TEMP_DELTA.to_unit([radiant_clg_dsgn_sup_wtr_temp_delt_r], 'dC', 'dF')[0]
    chilled_water_loop.sizingPlant().setDesignLoopExitTemperature(
        radiant_clg_dsgn_sup_wtr_temp_c)
    chilled_water_loop.sizingPlant().setLoopDesignTemperatureDifference(
        radiant_clg_dsgn_sup_wtr_temp_delt_k)
    chw_sch_name = '{} Temp - {}F'.format(
        chilled_water_loop.nameString(), round(radiant_clg_dsgn_sup_wtr_temp_f))
    chw_temp_sch = create_constant_schedule_ruleset(
        model, radiant_clg_dsgn_sup_wtr_temp_c, name=chw_sch_name,
        schedule_type_limit='Temperature')
    for spm in chilled_water_loop.supplyOutletNode().setpointManagers():
        if spm.to_SetpointManagerScheduled().is_initialized():
            spm = spm.to_SetpointManagerScheduled().get()
            spm.setSchedule(chw_temp_sch)

    # default temperature controls for radiant system
    zn_radiant_htg_dsgn_temp_f = 68.0
    zn_radiant_htg_dsgn_temp_c = \
        TEMPERATURE.to_unit([zn_radiant_htg_dsgn_temp_f], 'C', 'F')[0]
    zn_radiant_clg_dsgn_temp_f = 74.0
    zn_radiant_clg_dsgn_temp_c = \
        TEMPERATURE.to_unit([zn_radiant_clg_dsgn_temp_f], 'C', 'F')[0]

    htg_sch_name = 'Zone Radiant Loop Heating Threshold Temperature Schedule ' \
        '- {}F'.format(round(zn_radiant_htg_dsgn_temp_f))
    htg_control_temp_sch = create_constant_schedule_ruleset(
        model, zn_radiant_htg_dsgn_temp_c, name=htg_sch_name,
        schedule_type_limit='Temperature')
    clg_sch_name = 'Zone Radiant Loop Cooling Threshold Temperature Schedule ' \
        '- {}F'.format(round(zn_radiant_clg_dsgn_temp_f))
    clg_control_temp_sch = create_constant_schedule_ruleset(
        model, zn_radiant_clg_dsgn_temp_c, name=clg_sch_name,
        schedule_type_limit='Temperature')
    throttling_range_f = 4.0  # 2 degF on either side of control temperature
    throttling_range_c = TEMP_DELTA.to_unit([throttling_range_f], 'dC', 'dF')[0]

    # create preset availability schedule for radiant loop
    radiant_avail_sch = openstudio_model.ScheduleRuleset(model)
    radiant_avail_sch.setName('Radiant System Availability Schedule')

    if not radiant_lockout:
        radiant_availability_type = radiant_availability_type.lower()
        if radiant_availability_type == 'all_day':
            start_hour = 24
            start_minute = 0
            end_hour = 24
            end_minute = 0
        elif radiant_availability_type == 'afternoon_shutoff':
            start_hour = 15
            start_minute = 0
            end_hour = 22
            end_minute = 0
        elif radiant_availability_type == 'precool':
            start_hour = 10
            start_minute = 0
            end_hour = 22
            end_minute = 0
        elif radiant_availability_type == 'occupancy':
            start_hour = model_occ_hr_end.to_i
            start_minute = int((model_occ_hr_end % 1) * 60)
            end_hour = model_occ_hr_start.to_i
            end_minute = int((model_occ_hr_start % 1) * 60)
        else:
            msg = 'Unsupported radiant availability preset "{}"' \
                ' Defaulting to all day operation.'.format(radiant_availability_type)
            print(msg)
            start_hour = 24
            start_minute = 0
            end_hour = 24
            end_minute = 0

    # create custom availability schedule for radiant loop
    if radiant_lockout:
        start_hour = int(radiant_lockout_start_time)
        start_minute = int((radiant_lockout_start_time % 1) * 60)
        end_hour = radiant_lockout_end_time.to_i
        end_minute = int((radiant_lockout_end_time % 1) * 60)

    # create availability schedules
    if end_hour > start_hour:
        radiant_avail_sch.defaultDaySchedule().addValue(
            openstudio.Time(0, start_hour, start_minute, 0), 1.0)
        radiant_avail_sch.defaultDaySchedule().addValue(
            openstudio.Time(0, end_hour, end_minute, 0), 0.0)
        if end_hour < 24:
            radiant_avail_sch.defaultDaySchedule().addValue(
                openstudio.Time(0, 24, 0, 0), 1.0)
    elif start_hour > end_hour:
        radiant_avail_sch.defaultDaySchedule().addValue(
            openstudio.Time(0, end_hour, end_minute, 0), 0.0)
        radiant_avail_sch.defaultDaySchedule().addValue(
            openstudio.Time(0, start_hour, start_minute, 0), 1.0)
        if start_hour < 24:
            radiant_avail_sch.defaultDaySchedule().addValue(
                openstudio.Time(0, 24, 0, 0), 0.0)
    else:
        radiant_avail_sch.defaultDaySchedule().addValue(
            openstudio.Time(0, 24, 0, 0), 1.0)

    # add supply water temperature control if enabled
    if plant_supply_water_temperature_control:
        # add supply water temperature for heating plant loop
        model_add_plant_supply_water_temperature_control(
            model, hot_water_loop,
            control_strategy=plant_supply_water_temperature_control_strategy,
            sp_at_oat_low=hwsp_at_oat_low, oat_low=hw_oat_low,
            sp_at_oat_high=hwsp_at_oat_high, oat_high=hw_oat_high,
            thermal_zones=thermal_zones)

        # add supply water temperature for cooling plant loop
        model_add_plant_supply_water_temperature_control(
            model, chilled_water_loop,
            control_strategy=plant_supply_water_temperature_control_strategy,
            sp_at_oat_low=chwsp_at_oat_low, oat_low=chw_oat_low,
            sp_at_oat_high=chwsp_at_oat_high, oat_high=chw_oat_high,
            thermal_zones=thermal_zones)

    # make a low temperature radiant loop for each zone
    radiant_loops = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        if ':' in zone_name:
            msg = 'Thermal zone "{}" has a restricted character ":" in the name and ' \
                'will not work with some EMS and output reporting objects. '\
                'Please rename the zone'.format(zone_name)
            print(msg)

        # assign internal source construction to floors in zone
        srf_count = 0
        for space in zone.spaces():
            surfaces = space.surfaces() if sys.version_info >= (3, 0) else space.surfaces
            for surface in surfaces:
                if surface.isAirWall():
                    continue
                elif radiant_type == 'floor':
                    if surface.surfaceType() == 'Floor':
                        srf_count += 1
                        if surface.outsideBoundaryCondition() == 'Ground':
                            surface.setConstruction(radiant_ground_slab_construction)
                        elif surface.outsideBoundaryCondition() == 'Outdoors':
                            surface.setConstruction(radiant_exterior_slab_construction)
                        else:  # interior floor
                            surface.setConstruction(radiant_interior_floor_slab_construction)
                elif radiant_type == 'ceiling':
                    if surface.surfaceType() == 'RoofCeiling':
                        srf_count += 1
                        if surface.outsideBoundaryCondition == 'Outdoors':
                            surface.setConstruction(radiant_ceiling_slab_construction)
                        else:  # interior ceiling
                            surface.setConstruction(radiant_interior_ceiling_slab_construction)
                elif radiant_type == 'ceilingmetalpanel':
                    if surface.surfaceType() == 'RoofCeiling':
                        srf_count += 1
                        if surface.outsideBoundaryCondition() == 'Outdoors':
                            surface.setConstruction(radiant_ceiling_metal_construction)
                        else:  # interior ceiling
                            surface.setConstruction(radiant_interior_ceiling_metal_construction)
                elif radiant_type == 'floorwithhardwood':
                    if surface.surfaceType() == 'Floor':
                        srf_count += 1
                        if surface.outsideBoundaryCondition() == 'Ground':
                            surface.setConstruction(radiant_ground_wood_construction)
                        elif surface.outsideBoundaryCondition() == 'Outdoors':
                            surface.setConstruction(radiant_exterior_wood_construction)
                        else:  # interior floor
                            surface.setConstruction(radiant_interior_wood_floor_construction)

        # ignore the Zone if it has not thermally active Faces
        if srf_count == 0:
            continue

        # create radiant coils
        radiant_loop_htg_coil = openstudio_model.CoilHeatingLowTempRadiantVarFlow(
            model, htg_control_temp_sch)
        radiant_loop_htg_coil.setName('{} Radiant Loop Heating Coil'.format(zone_name))
        radiant_loop_htg_coil.setHeatingControlThrottlingRange(throttling_range_c)
        hot_water_loop.addDemandBranchForComponent(radiant_loop_htg_coil)

        radiant_loop_clg_coil = openstudio_model.CoilCoolingLowTempRadiantVarFlow(
            model, clg_control_temp_sch)
        radiant_loop_clg_coil.setName('{} Radiant Loop Cooling Coil'.format(zone_name))
        radiant_loop_clg_coil.setCoolingControlThrottlingRange(throttling_range_c)
        chilled_water_loop.addDemandBranchForComponent(radiant_loop_clg_coil)

        radiant_loop = openstudio_model.ZoneHVACLowTempRadiantVarFlow(
            model, radiant_avail_sch, radiant_loop_htg_coil, radiant_loop_clg_coil)

        # radiant loop surfaces
        radiant_loop.setName('{} Radiant Loop'.format(zone_name))
        if radiant_type == 'floor':
            radiant_loop.setRadiantSurfaceType('Floors')
        elif radiant_type == 'ceiling':
            radiant_loop.setRadiantSurfaceType('Ceilings')
        elif radiant_type == 'ceilingmetalpanel':
            radiant_loop.setRadiantSurfaceType('Ceilings')
        elif radiant_type == 'floorwithhardwood':
            radiant_loop.setRadiantSurfaceType('Floors')

        # radiant loop layout details
        radiant_loop.setHydronicTubingInsideDiameter(0.015875)  # 5/8 in. ID, 3/4 in. OD
        radiant_loop.setNumberofCircuits('CalculateFromCircuitLength')
        radiant_loop.setCircuitLength(106.7)

        # radiant loop temperature controls
        radiant_loop.setTemperatureControlType(radiant_temperature_control_type)

        # radiant loop setpoint temperature response
        radiant_loop.setSetpointControlType(radiant_setpoint_control_type)
        radiant_loop.addToThermalZone(zone)
        radiant_loops.append(radiant_loop)

        # rename nodes before adding EMS code
        rename_plant_loop_nodes(model)

        # set radiant loop controls
        control_strategy = control_strategy.lower()
        if control_strategy == 'proportional_control':
            # slab setpoint varies based on previous day zone conditions
            model_add_radiant_proportional_controls(
                model, zone, radiant_loop,
                radiant_temperature_control_type=radiant_temperature_control_type,
                use_zone_occupancy_for_control=use_zone_occupancy_for_control,
                occupied_percentage_threshold=occupied_percentage_threshold,
                model_occ_hr_start=model_occ_hr_start, model_occ_hr_end=model_occ_hr_end,
                proportional_gain=proportional_gain, switch_over_time=switch_over_time)
        elif control_strategy == 'oa_based_control':
            # slab setpoint varies based on outdoor weather
            model_add_radiant_basic_controls(
                model, zone, radiant_loop,
                radiant_temperature_control_type=radiant_temperature_control_type,
                slab_setpoint_oa_control=True, switch_over_time=switch_over_time,
                slab_sp_at_oat_low=slab_sp_at_oat_low, slab_oat_low=slab_oat_low,
                slab_sp_at_oat_high=slab_sp_at_oat_high, slab_oat_high=slab_oat_high)
        elif control_strategy == 'constant_control':
            # constant slab setpoint control
            model_add_radiant_basic_controls(
                model, zone, radiant_loop,
                radiant_temperature_control_type=radiant_temperature_control_type,
                slab_setpoint_oa_control=False, switch_over_time=switch_over_time,
                slab_sp_at_oat_low=slab_sp_at_oat_low, slab_oat_low=slab_oat_low,
                slab_sp_at_oat_high=slab_sp_at_oat_high, slab_oat_high=slab_oat_high)
    return radiant_loops


def model_add_window_ac(model, thermal_zones):
    """Adds a window air conditioner to each zone. Code adapted from.

    https://github.com/NREL/OpenStudio-BEopt/blob/master/measures/
    ResidentialHVACRoomAirConditioner/measure.rb

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to add window AC units to.
    """
    # Defaults
    eer = 8.5  # Btu/W-h
    cop = POWER.to_unit([eer], 'W', 'Btu/h')[0]
    # ratio of the sensible portion of the load to the total load:
    shr = 0.65  # sensible heat ratio at the nominal rated capacity

    acs = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        clg_coil = create_coil_cooling_dx_single_speed(
            model, name='{} Window AC Cooling Coil'.format(zone_name),
            type='Window AC', cop=cop)
        clg_coil.setRatedSensibleHeatRatio(shr)
        if model.version() < openstudio.VersionString('3.5.0'):
            clg_coil.setRatedEvaporatorFanPowerPerVolumeFlowRate(773.3)
        else:
            clg_coil.setRatedEvaporatorFanPowerPerVolumeFlowRate2017(773.3)
        clg_coil.setEvaporativeCondenserEffectiveness(0.9)
        clg_coil.setMaximumOutdoorDryBulbTemperatureForCrankcaseHeaterOperation(10)
        clg_coil.setBasinHeaterSetpointTemperature(2)

        fan = create_fan_by_name(
            model, 'Window_AC_Supply_Fan',
            fan_name='{} Window AC Supply Fan'.format(zone_name),
            end_use_subcategory='Window AC Fans')
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        htg_coil = create_coil_heating_electric(
            model, name='{} Window AC Always Off Htg Coil'.format(zone_name),
            schedule=model.alwaysOffDiscreteSchedule(), nominal_capacity=0)
        ptac = openstudio_model.ZoneHVACPackagedTerminalAirConditioner(
            model, model.alwaysOnDiscreteSchedule(), fan, htg_coil, clg_coil)
        ptac.setName('{} Window AC'.format(zone_name))
        ptac.setSupplyAirFanOperatingModeSchedule(model.alwaysOffDiscreteSchedule())
        ptac.addToThermalZone(zone)
        acs.append(ptac)

    return acs


def model_add_furnace_central_ac(
        model, thermal_zones, heating=True, cooling=False, ventilation=False,
        heating_type='NaturalGas'):
    """Adds a forced air furnace or central AC to each zone.

    Default is a forced air furnace without outdoor air. Code adapted from.

    https://github.com/NREL/OpenStudio-BEopt/blob/master/measures/
    ResidentialHVACFurnaceFuel/measure.rb

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            add furnace to.
        heating: [Boolean] if true, the unit will include a heating coil.
        cooling: [Boolean] if true, the unit will include a DX cooling coil.
        ventilation: [Boolean] if true, the unit will include an OA intake.
        heating_type: [String] valid choices are NaturalGas, Gas, Electricity, Electric.
    """

    if heating and cooling:
        equip_name = 'Central Heating and AC'
    elif heating and not cooling:
        equip_name = 'Furnace'
    elif cooling and not heating:
        equip_name = 'Central AC'
    else:
        msg = 'Heating and cooling both disabled, not a valid Furnace or ' \
            'Central AC selection, no equipment was added.'
        print(msg)
        return None

    # defaults
    afue = 0.78
    # seer = 13.0
    eer = 11.1
    shr = 0.73
    ac_w_per_cfm = 0.365
    crank_case_heat_w = 0.0
    crank_case_max_temp_f = 55.0

    furnaces = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        air_loop = openstudio_model.AirLoopHVAC(model)
        air_loop.setName('{} {}'.format(zone_name, equip_name))
        loop_name = air_loop.nameString()

        # default design temperatures across all air loops
        dsgn_temps = standard_design_sizing_temperatures()

        # adjusted temperatures for furnace_central_ac
        dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
        dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
            TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
        dsgn_temps['htg_dsgn_sup_air_temp_f'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_f']
        dsgn_temps['htg_dsgn_sup_air_temp_c'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_c']

        # default design settings used across all air loops
        sizing_system = adjust_sizing_system(
            air_loop, dsgn_temps, sizing_option='NonCoincident')
        sizing_system.setAllOutdoorAirinCooling(True)
        sizing_system.setAllOutdoorAirinHeating(True)

        # create heating coil
        htg_coil = None
        if heating:
            heating_type = 'NaturalGas' if heating_type is None else heating_type
            if heating_type in ('NaturalGas', 'Gas'):
                htg_coil = create_coil_heating_gas(
                    model, name='{} Heating Coil'.format(loop_name), efficiency=afue)
            else:  # electric coil
                htg_coil = create_coil_heating_electric(
                    model, name='{} Heating Coil'.format(loop_name))

        # create cooling coil
        clg_coil = None
        if cooling:
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} Cooling Coil'.format(loop_name),
                type='Residential Central AC')
            clg_coil.setRatedSensibleHeatRatio(shr)
            clg_coil.setRatedCOP(eer_to_cop_no_fan(eer))
            ac_w_per_mps = ac_w_per_cfm / FLOW_RATE.to_unit([1.0], 'm3/s', 'cfm')[0]
            if model.version() < openstudio.VersionString('3.5.0'):
                clg_coil.setRatedEvaporatorFanPowerPerVolumeFlowRate(ac_w_per_mps)
            else:
                clg_coil.setRatedEvaporatorFanPowerPerVolumeFlowRate2017(ac_w_per_mps)
            clg_coil.setNominalTimeForCondensateRemovalToBegin(1000.0)
            clg_coil.setRatioOfInitialMoistureEvaporationRateAndSteadyStateLatentCapacity(1.5)
            clg_coil.setMaximumCyclingRate(3.0)
            clg_coil.setLatentCapacityTimeConstant(45.0)
            clg_coil.setCondenserType('AirCooled')
            clg_coil.setCrankcaseHeaterCapacity(crank_case_heat_w)
            clg_coil.setMaximumOutdoorDryBulbTemperatureForCrankcaseHeaterOperation(
                TEMPERATURE.to_unit([crank_case_max_temp_f], 'C', 'F')[0])

        # create fan
        fan = create_fan_by_name(
            model, 'Residential_HVAC_Fan', fan_name='{} Supply Fan'.format(loop_name),
            end_use_subcategory='Residential HVAC Fans')
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        if ventilation:
            # create outdoor air intake
            oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
            oa_intake_controller.setName('{} OA Controller'.format(loop_name))
            oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
            oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
            oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(
                model, oa_intake_controller)
            oa_intake.setName('{} OA System'.format(loop_name))
            oa_intake.addToNode(air_loop.supplyInletNode())

        # create unitary system (holds the coils and fan)
        unitary = openstudio_model.AirLoopHVACUnitarySystem(model)
        unitary.setName('{} Unitary System'.format(loop_name))
        unitary.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
        unitary.setMaximumSupplyAirTemperature(dsgn_temps['zn_htg_dsgn_sup_air_temp_c'])
        unitary.setControllingZoneorThermostatLocation(zone)
        unitary.addToNode(air_loop.supplyInletNode())

        # set flow rates during different conditions
        if not heating:
            unitary.setSupplyAirFlowRateDuringHeatingOperation(0.0)
        if not cooling:
            unitary.setSupplyAirFlowRateDuringCoolingOperation(0.0)
        if not ventilation:
            unitary.setSupplyAirFlowRateWhenNoCoolingorHeatingisRequired(0.0)

        # attach the coils and fan
        if htg_coil is not None:
            unitary.setHeatingCoil(htg_coil)
        if clg_coil is not None:
            unitary.setCoolingCoil(clg_coil)
        unitary.setSupplyFan(fan)
        unitary.setFanPlacement('BlowThrough')
        unitary.setSupplyAirFanOperatingModeSchedule(model.alwaysOffDiscreteSchedule())

        # create a diffuser
        if model.version() < openstudio.VersionString('2.7.0'):
            diffuser = openstudio_model.AirTerminalSingleDuctUncontrolled(
                model, model.alwaysOnDiscreteSchedule())
        else:
            diffuser = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                model, model.alwaysOnDiscreteSchedule())
        diffuser.setName('{} Direct Air'.format(zone_name))
        air_loop.multiAddBranchForZone(zone, diffuser.to_HVACComponent().get())

        furnaces.append(air_loop)

    return furnaces


def model_add_central_air_source_heat_pump(
        model, thermal_zones, heating=True, cooling=True, ventilation=False):
    """Adds an air source heat pump to each zone. Code adapted from.

    https://github.com/NREL/OpenStudio-BEopt/blob/master/measures/
    ResidentialHVACAirSourceHeatPumpSingleSpeed/measure.rb

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones
            to add fan coil units to.
        heating: [Boolean] if true, the unit will include a heating coil.
        cooling: [Boolean] if true, the unit will include a DX cooling coil.
        ventilation: [Boolean] if true, the unit will include an OA intake.
    """
    # defaults
    hspf = 7.7
    # seer = 13.0
    # eer = 11.4
    cop = 3.05
    shr = 0.73
    ac_w_per_cfm = 0.365
    min_hp_oat_f = 0.0
    crank_case_heat_w = 0.0
    crank_case_max_temp_f = 55

    # default design temperatures across all air loops
    dsgn_temps = standard_design_sizing_temperatures()

    # adjusted temperatures for furnace_central_ac
    dsgn_temps['zn_htg_dsgn_sup_air_temp_f'] = 122.0
    dsgn_temps['zn_htg_dsgn_sup_air_temp_c'] = \
        TEMPERATURE.to_unit([dsgn_temps['zn_htg_dsgn_sup_air_temp_f']], 'C', 'F')[0]
    dsgn_temps['htg_dsgn_sup_air_temp_f'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_f']
    dsgn_temps['htg_dsgn_sup_air_temp_c'] = dsgn_temps['zn_htg_dsgn_sup_air_temp_c']

    hps = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        air_loop = openstudio_model.AirLoopHVAC(model)
        air_loop.setName('{} Central Air Source HP'.format(zone_name))
        loop_name = air_loop.nameString()

        # default design settings used across all air loops
        sizing_system = adjust_sizing_system(
            air_loop, dsgn_temps, sizing_option='NonCoincident')
        sizing_system.setAllOutdoorAirinCooling(True)
        sizing_system.setAllOutdoorAirinHeating(True)

        # create heating coil
        htg_coil = None
        supplemental_htg_coil = None
        if heating:
            htg_coil = create_coil_heating_dx_single_speed(
                model, name='{} heating coil'.format(loop_name),
                type='Residential Central Air Source HP', cop=hspf_to_cop_no_fan(hspf))
            if model.version() < openstudio.VersionString('3.5.0'):
                htg_coil.setRatedSupplyFanPowerPerVolumeFlowRate(
                    ac_w_per_cfm / FLOW_RATE.to_unit([1.0], 'm3/s', 'cfm')[0])
            else:
                htg_coil.setRatedSupplyFanPowerPerVolumeFlowRate2017(
                    ac_w_per_cfm / FLOW_RATE.to_unit([1.0], 'm3/s', 'cfm')[0])
            htg_coil.setMinimumOutdoorDryBulbTemperatureforCompressorOperation(
                TEMPERATURE.to_unit([min_hp_oat_f], 'C', 'F')[0])
            htg_coil.setMaximumOutdoorDryBulbTemperatureforDefrostOperation(
                TEMPERATURE.to_unit([40.0], 'C', 'F')[0])
            htg_coil.setCrankcaseHeaterCapacity(crank_case_heat_w)
            htg_coil.setMaximumOutdoorDryBulbTemperatureforCrankcaseHeaterOperation(
                TEMPERATURE.to_unit([crank_case_max_temp_f], 'C', 'F')[0])
            htg_coil.setDefrostStrategy('ReverseCycle')
            htg_coil.setDefrostControl('OnDemand')
            htg_coil.resetDefrostTimePeriodFraction()

            # create supplemental heating coil
            supplemental_htg_coil = create_coil_heating_electric(
                model, name='{} Supplemental Htg Coil'.format(loop_name))

        # create cooling coil
        clg_coil = None
        if cooling:
            clg_coil = create_coil_cooling_dx_single_speed(
                model, name='{} Cooling Coil'.format(loop_name),
                type='Residential Central ASHP', cop=cop)
            clg_coil.setRatedSensibleHeatRatio(shr)
            ac_w_per_mps = ac_w_per_cfm / FLOW_RATE.to_unit([1.0], 'm3/s', 'cfm')[0]
            if model.version() < openstudio.VersionString('3.5.0'):
                clg_coil.setRatedEvaporatorFanPowerPerVolumeFlowRate(ac_w_per_mps)
            else:
                clg_coil.setRatedEvaporatorFanPowerPerVolumeFlowRate2017(ac_w_per_mps)
            clg_coil.setNominalTimeForCondensateRemovalToBegin(1000.0)
            clg_coil.setRatioOfInitialMoistureEvaporationRateAndSteadyStateLatentCapacity(1.5)
            clg_coil.setMaximumCyclingRate(3.0)
            clg_coil.setLatentCapacityTimeConstant(45.0)
            clg_coil.setCondenserType('AirCooled')
            clg_coil.setCrankcaseHeaterCapacity(crank_case_heat_w)
            clg_coil.setMaximumOutdoorDryBulbTemperatureForCrankcaseHeaterOperation(
                TEMPERATURE.to_unit([crank_case_max_temp_f], 'C', 'F')[0])

        # create fan
        fan = create_fan_by_name(
            model, 'Residential_HVAC_Fan', fan_name='{} Supply Fan'.format(loop_name),
            end_use_subcategory='Residential HVAC Fans')
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        # create outdoor air intake
        if ventilation:
            oa_intake_controller = openstudio_model.ControllerOutdoorAir(model)
            oa_intake_controller.setName('{} OA Controller'.format(loop_name))
            oa_intake_controller.autosizeMinimumOutdoorAirFlowRate()
            oa_intake_controller.resetEconomizerMinimumLimitDryBulbTemperature()
            oa_intake = openstudio_model.AirLoopHVACOutdoorAirSystem(
                model, oa_intake_controller)
            oa_intake.setName('{} OA System'.format(loop_name))
            oa_intake.addToNode(air_loop.supplyInletNode())

        # create unitary system (holds the coils and fan)
        unitary = openstudio_model.AirLoopHVACUnitarySystem(model)
        unitary.setName('{} Unitary System'.format(loop_name))
        unitary.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())
        # higher temp for supplemental heat as to not severely limit its use,
        # resulting in unmet hours
        unitary.setMaximumSupplyAirTemperature(TEMPERATURE.to_unit([170.0], 'C', 'F')[0])
        unitary.setMaximumOutdoorDryBulbTemperatureforSupplementalHeaterOperation(
            TEMPERATURE.to_unit([40.0], 'C', 'F')[0])
        unitary.setControllingZoneorThermostatLocation(zone)
        unitary.addToNode(air_loop.supplyInletNode())

        # set flow rates during different conditions
        if not ventilation:
            unitary.setSupplyAirFlowRateWhenNoCoolingorHeatingisRequired(0.0)

        # attach the coils and fan
        if htg_coil is not None:
            unitary.setHeatingCoil(htg_coil)
        if clg_coil is not None:
            unitary.setCoolingCoil(clg_coil)
        if supplemental_htg_coil is not None:
            unitary.setSupplementalHeatingCoil(supplemental_htg_coil)
        unitary.setSupplyFan(fan)
        unitary.setFanPlacement('BlowThrough')
        unitary.setSupplyAirFanOperatingModeSchedule(model.alwaysOffDiscreteSchedule())

        # create a diffuser
        if model.version() < openstudio.VersionString('2.7.0'):
            diffuser = openstudio_model.AirTerminalSingleDuctUncontrolled(
                model, model.alwaysOnDiscreteSchedule())
        else:
            diffuser = openstudio_model.AirTerminalSingleDuctConstantVolumeNoReheat(
                model, model.alwaysOnDiscreteSchedule())
        diffuser.setName('{} Direct Air'.format(zone_name))
        air_loop.multiAddBranchForZone(zone, diffuser.to_HVACComponent().get())

        hps.append(air_loop)

    return hps


def model_add_water_source_hp(model, thermal_zones, condenser_loop, ventilation=True):
    """Adds zone level water-to-air heat pumps for each zone.

    Args:
        model [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones [Array<OpenStudio::Model::ThermalZone>] array of zones
            served by heat pumps.
        condenser_loop [OpenStudio::Model::PlantLoop] the condenser loop for
            the heat pumps.
        ventilation [Boolean] if true, ventilation will be supplied through the unit.
            If false, no ventilation will be supplied through the unit, with
            the expectation that it will be provided by a DOAS or separate system.
    """
    water_to_air_hp_systems = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        supplemental_htg_coil = create_coil_heating_electric(
            model, name='{} Supplemental Htg Coil'.format(zone_name))
        htg_coil = create_coil_heating_water_to_air_heat_pump_equation_fit(
            model, condenser_loop, name='{} Water-to-Air HP Htg Coil'.format(zone_name))
        clg_coil = create_coil_cooling_water_to_air_heat_pump_equation_fit(
            model, condenser_loop, name='{} Water-to-Air HP Clg Coil'.format(zone_name))

        # add fan
        fan = create_fan_by_name(model, 'WSHP_Fan',
                                 fan_name='{} WSHP Fan'.format(zone_name))
        fan.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

        water_to_air_hp_system = openstudio_model.ZoneHVACWaterToAirHeatPump(
            model, model.alwaysOnDiscreteSchedule(), fan, htg_coil, clg_coil,
            supplemental_htg_coil)
        water_to_air_hp_system.setName('{} WSHP'.format(zone_name))
        if not ventilation:
            water_to_air_hp_system.setOutdoorAirFlowRateDuringHeatingOperation(0.0)
            water_to_air_hp_system.setOutdoorAirFlowRateDuringCoolingOperation(0.0)
            water_to_air_hp_system.setOutdoorAirFlowRateWhenNoCoolingorHeatingisNeeded(0.0)
        water_to_air_hp_system.addToThermalZone(zone)

        water_to_air_hp_systems.append(water_to_air_hp_system)

    return water_to_air_hp_systems


def model_add_zone_erv(model, thermal_zones):
    """Adds zone level ERVs for each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            add heat pumps to.
    """
    ervs = []
    for zone in thermal_zones:
        zone_name = zone.nameString()
        # Determine the OA requirement for this zone
        min_oa_flow_m3_per_s_per_m2 = thermal_zone_get_outdoor_airflow_rate_per_area(zone)
        supply_fan = create_fan_by_name(model, 'ERV_Supply_Fan',
                                        fan_name='{} ERV Supply Fan'.format(zone_name))
        fan_change_impeller_efficiency(supply_fan, impeller_eff=0.55)
        exhaust_fan = create_fan_by_name(model, 'ERV_Supply_Fan',
                                         fan_name='{} ERV Exhaust Fan'.format(zone_name))
        fan_change_impeller_efficiency(exhaust_fan, impeller_eff=0.55)

        erv_controller = openstudio_model.ZoneHVACEnergyRecoveryVentilatorController(model)
        erv_controller.setName('{} ERV Controller'.format(zone_name))

        heat_exchanger = create_hx_air_to_air_sensible_and_latent(
            model, name='{} ERV HX'.format(zone_name), type='Plate',
            economizer_lockout=False, supply_air_outlet_temperature_control=False,
            sensible_heating_100_eff=0.76, latent_heating_100_eff=0.68,
            sensible_cooling_100_eff=0.76, latent_cooling_100_eff=0.68,)

        zone_hvac = openstudio_model.ZoneHVACEnergyRecoveryVentilator(
            model, heat_exchanger, supply_fan, exhaust_fan)
        zone_hvac.setName('{} ERV'.format(zone_name))
        zone_hvac.setVentilationRateperUnitFloorArea(min_oa_flow_m3_per_s_per_m2)
        zone_hvac.setController(erv_controller)
        zone_hvac.addToThermalZone(zone)

        # ensure the ERV takes priority, so ventilation load is included when
        # treated by other zonal systems
        zone.setCoolingPriority(zone_hvac.to_ModelObject().get(), 1)
        zone.setHeatingPriority(zone_hvac.to_ModelObject().get(), 1)

        # set the cooling and heating fraction to zero so that the ERV does
        # not try to meet the heating or cooling load.
        if model.version() >= openstudio.VersionString('2.8.0'):
            zone.setSequentialCoolingFraction(zone_hvac.to_ModelObject().get(), 0.0)
            zone.setSequentialHeatingFraction(zone_hvac.to_ModelObject().get(), 0.0)

        # Calculate ERV SAT during sizing periods
        # Standard rating conditions based on AHRI Std 1060 - 2013
        # heating design
        oat_f = 35.0
        return_air_f = 70.0
        eff = heat_exchanger.sensibleEffectivenessat100HeatingAirFlow()
        coldest_erv_supply_f = oat_f - (eff * (oat_f - return_air_f))
        coldest_erv_supply_c = TEMPERATURE.to_unit([coldest_erv_supply_f], 'C', 'F')[0]

        # cooling design
        oat_f = 95.0
        return_air_f = 75.0
        eff = heat_exchanger.sensibleEffectivenessat100CoolingAirFlow()
        hottest_erv_supply_f = oat_f - (eff * (oat_f - return_air_f))
        hottest_erv_supply_c = TEMPERATURE.to_unit([hottest_erv_supply_f], 'C', 'F')[0]

        # Ensure that zone sizing accounts for OA from ERV
        sizing_zone = zone.sizingZone()
        sizing_zone.setAccountforDedicatedOutdoorAirSystem(True)
        sizing_zone.setDedicatedOutdoorAirSystemControlStrategy('NeutralSupplyAir')
        sizing_zone.setDedicatedOutdoorAirLowSetpointTemperatureforDesign(
            coldest_erv_supply_c)
        sizing_zone.setDedicatedOutdoorAirHighSetpointTemperatureforDesign(
            hottest_erv_supply_c)

        ervs.append(zone_hvac)

    return ervs


def model_add_residential_erv(model, thermal_zones):
    """Add residential zone level ERVs for each zone.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            add heat pumps to.
    """
    # consider implementing the full method in openstudio-standards
    # this method has many different criteria based on the version of AHSRAE 90.1
    # is also varies with climate zone
    return model_add_zone_erv(model, thermal_zones)


def model_add_residential_ventilator(model, thermal_zones):
    """Add a residential ventilation.

    Ventilators are standalone unit ventilation and zone exhaust that operates
    to provide OA, used in conjuction with a system that having mechanical
    cooling and a heating coil.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            enable ideal air loads.
    """
    unit_ventilators = []
    for thermal_zone in thermal_zones:
        zone_name = thermal_zone.nameString()
        min_oa_flow_m3 = thermal_zone_get_outdoor_airflow_rate(thermal_zone)
        # Fan power with no energy recovery = 0.806 W/cfm
        supply_fan = create_fan_by_name(
            model, 'ERV_Supply_Fan',
            fan_name='{} Ventilator Supply Fan'.format(zone_name))
        supply_fan.setMotorEfficiency(0.48)
        supply_fan.setFanTotalEfficiency(0.303158)
        supply_fan.setPressureRise(233.6875)

        unit_ventilator = openstudio_model.ZoneHVACUnitVentilator(model, supply_fan)
        unit_ventilator.setName('{} Unit Ventilator'.format(zone_name))
        unit_ventilator.addToThermalZone(thermal_zone)

        fan_zone_exhaust = create_fan_by_name(
            model, 'ERV_Exhaust_fan', fan_name='{} Exhaust Fan'.format(zone_name))
        fan_zone_exhaust.setMotorEfficiency(0.48)
        fan_zone_exhaust.setFanEfficiency(0.303158)
        fan_zone_exhaust.setPressureRise(233.6875)

        # Set OA requirements; Assumes a default of 55 cfm
        unit_ventilator.setMaximumSupplyAirFlowRate(min_oa_flow_m3)
        fan_zone_exhaust.setMaximumFlowRate(min_oa_flow_m3)

        # Ensure the unit ventilator takes priority, so ventilation load is
        # included when treated by other zonal systems
        thermal_zone.setCoolingPriority(unit_ventilator.to_ModelObject.get, 1)
        thermal_zone.setHeatingPriority(unit_ventilator.to_ModelObject.get, 1)

        unit_ventilators.append(unit_ventilator)

    return unit_ventilators


def model_add_waterside_economizer(
        model, chilled_water_loop, condenser_water_loop, integrated=True):
    """Adds a waterside economizer to the chilled water and condenser loop.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        integrated [Boolean] when set to true, models an integrated waterside economizer.
            Integrated - in series with chillers, can run simultaneously with chillers
            Non-Integrated - in parallel with chillers, chillers locked out during operation
    """
    # make a new heat exchanger
    heat_exchanger = openstudio_model.HeatExchangerFluidToFluid(model)
    heat_exchanger.setHeatExchangeModelType('CounterFlow')
    # zero degree minimum necessary to allow both economizer and heat exchanger
    # to operate in both integrated and non-integrated archetypes
    # possibly results from an EnergyPlus issue that didn't get resolved correctly
    # https://github.com/NREL/EnergyPlus/issues/5626
    heat_exchanger.setMinimumTemperatureDifferencetoActivateHeatExchanger(0.0)
    heat_exchanger.setHeatTransferMeteringEndUseType('FreeCooling')
    o_min_tl = TEMPERATURE.to_unit([35.0], 'C', 'F')[0]
    heat_exchanger.setOperationMinimumTemperatureLimit(o_min_tl)
    o_max_tl = TEMPERATURE.to_unit([72.0], 'C', 'F')[0]
    heat_exchanger.setOperationMaximumTemperatureLimit(o_max_tl)
    heat_exchanger.setAvailabilitySchedule(model.alwaysOnDiscreteSchedule())

    # get the chillers on the chilled water loop
    chillers = chilled_water_loop.supplyComponents(
        'OS:Chiller:Electric:EIR'.to_IddObjectType())

    if integrated:
        if os_vector_len(chillers) == 0:
            msg = 'No chillers were found on {}; only modeling waterside economizer'.format(
                chilled_water_loop.nameString())
            print(msg)

        # set methods for integrated heat exchanger
        heat_exchanger.setName('Integrated Waterside Economizer Heat Exchanger')
        heat_exchanger.setControlType('CoolingDifferentialOnOff')

        # add the heat exchanger to the chilled water loop upstream of the chiller
        heat_exchanger.addToNode(chilled_water_loop.supplyInletNode())

        # Copy the setpoint managers from the plant's supply outlet node
        # to the chillers and HX outlets.
        # This is necessary so that the correct type of operation scheme will be created.
        # Without this, OS will create an uncontrolled operation scheme
        # and the chillers will never run.
        chw_spms = chilled_water_loop.supplyOutletNode.setpointManagers()
        objs = []
        for obj in chillers:
            objs.append(obj.to_ChillerElectricEIR.get())
        objs.append(heat_exchanger)
        for obj in objs:
            outlet = obj.supplyOutletModelObject().get().to_Node().get()
            for spm in chw_spms:
                new_spm = spm.clone().to_SetpointManager().get()
                new_spm.addToNode(outlet)
    else:
        # non-integrated
        # if the heat exchanger can meet the entire load, the heat exchanger will run
        # and the chiller is disabled.
        # In E+, only one chiller can be tied to a given heat exchanger, so if you have
        # multiple chillers, they will cannot be tied to a single heat exchanger without EMS.
        chiller = None
        if os_vector_len(chillers) == 0:
            msg = 'No chillers were found on {}; cannot add a non-integrated ' \
                'waterside economizer.'.format(chilled_water_loop.nameString())
            print(msg)
            heat_exchanger.setControlType('CoolingSetpointOnOff')
        elif os_vector_len(chillers) > 1:
            chiller = chillers[0]
            msg = 'More than one chiller was found on {}. EnergyPlus only allows a ' \
                'single chiller to be interlocked with the HX.  Chiller {} was selected.' \
                ' Additional chillers will not be locked out during HX operation.'.format(
                    chilled_water_loop.nameString(), chiller.nameString())
            print(msg)
        else:  # 1 chiller
            chiller = chillers[0]
        chiller = chiller.to_ChillerElectricEIR().get()

        # set methods for non-integrated heat exchanger
        heat_exchanger.setName('Non-Integrated Waterside Economizer Heat Exchanger')
        heat_exchanger.setControlType('CoolingSetpointOnOffWithComponentOverride')

        # add the heat exchanger to a supply side branch of the chilled water loop
        # parallel with the chiller(s)
        chilled_water_loop.addSupplyBranchForComponent(heat_exchanger)

        # Copy the setpoint managers from the plant's supply outlet node to the HX outlet.
        # This is necessary so that the correct type of operation scheme will be created.
        # Without this, the HX will never run
        chw_spms = chilled_water_loop.supplyOutletNode().setpointManagers()
        outlet = heat_exchanger.supplyOutletModelObject().get().to_Node().get()
        for spm in chw_spms:
            new_spm = spm.clone().to_SetpointManager().get()
            new_spm.addToNode(outlet)

        # set the supply and demand inlet fields to interlock the heat exchanger with the chiller
        chiller_supply_inlet = chiller.supplyInletModelObject().get().to_Node().get()
        heat_exchanger.setComponentOverrideLoopSupplySideInletNode(chiller_supply_inlet)
        chiller_demand_inlet = chiller.demandInletModelObject().get().to_Node().get()
        heat_exchanger.setComponentOverrideLoopDemandSideInletNode(chiller_demand_inlet)

        # check if the chilled water pump is on a branch with the chiller.
        # if it is, move this pump before the splitter so that it can push water
        # through either the chiller or the heat exchanger.
        pumps_on_branches = []
        # search for constant and variable speed pumps between supply splitter and supply mixer.
        supply_comps = chilled_water_loop.supplyComponents(
            chilled_water_loop.supplySplitter(), chilled_water_loop.supplyMixer())
        for supply_comp in supply_comps:
            if supply_comp.to_PumpConstantSpeed().is_initialized():
                pumps_on_branches.append(supply_comp.to_PumpConstantSpeed().get())
            elif supply_comp.to_PumpVariableSpeed().is_initialized():
                pumps_on_branches.append(supply_comp.to_PumpVariableSpeed().get())
        # If only one pump is found, clone it, put the clone on the supply inlet node,
        # and delete the original pump.
        # If multiple branch pumps, clone the first pump found, add it to the inlet
        # of the heat exchanger, and warn user.
        if len(pumps_on_branches) == 1:
            pump = pumps_on_branches[0]
            pump_clone = pump.clone(model).to_StraightComponent().get()
            pump_clone.addToNode(chilled_water_loop.supplyInletNode())
            pump.remove()
        elif len(pumps_on_branches) > 1:
            hx_inlet_node = heat_exchanger.inletModelObject().get().to_Node().get()
            pump = pumps_on_branches[0]
            pump_clone = pump.clone(model).to_StraightComponent().get()
            pump_clone.addToNode(hx_inlet_node)

    # add heat exchanger to condenser water loop
    condenser_water_loop.addDemandBranchForComponent(heat_exchanger)

    # change setpoint manager on condenser water loop to allow waterside economizing
    dsgn_sup_wtr_temp_f = 42.0
    dsgn_sup_wtr_temp_c = TEMPERATURE.to_unit([42.0], 'C', 'F')[0]
    for spm in condenser_water_loop.supplyOutletNode().setpointManagers():
        if spm.to_SetpointManagerFollowOutdoorAirTemperature().is_initialized():
            spm = spm.to_SetpointManagerFollowOutdoorAirTemperature().get()
            spm.setMinimumSetpointTemperature(dsgn_sup_wtr_temp_c)
        elif spm.to_SetpointManagerScheduled().is_initialized():
            spm = spm.to_SetpointManagerScheduled().get()()
            cw_temp_sch = create_constant_schedule_ruleset(
                model, dsgn_sup_wtr_temp_c,
                name='{} Temp - {}F'.format(
                    chilled_water_loop.nameString(), int(dsgn_sup_wtr_temp_f)),
                schedule_type_limit='Temperature')
            spm.setSchedule(cw_temp_sch)
        else:
            msg = 'Condenser water loop {} setpoint manager {} is not a recognized ' \
                'setpoint manager type. Cannot change to account for the waterside ' \
                'economizer.'.format(condenser_water_loop.nameString(), spm.nameString())
            print(msg)

    return heat_exchanger


def model_add_zone_heat_cool_request_count_program(model, thermal_zones):
    """Make EMS program that will compare 'measured' zone air temperatures to setpoints.

    This can be used to determine if zone needs cooling or heating. Program will
    output the total zones needing heating and cooling and the their ratio using
    the total number of zones.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones to
            dictate cooling or heating mode of water plant.
    """
    # create container schedules to hold number of zones needing heating and cooling
    sch_zones_needing_heating = create_constant_schedule_ruleset(
        model, 0, name='Zones Needing Heating Count Schedule',
        schedule_type_limit='Dimensionless')
    zone_needing_heating_actuator = openstudio_model.EnergyManagementSystemActuator(
        sch_zones_needing_heating, 'Schedule:Year', 'Schedule Value')
    zone_needing_heating_actuator.setName('Zones_Needing_Heating')

    sch_zones_needing_cooling = create_constant_schedule_ruleset(
        model, 0, name='Zones Needing Cooling Count Schedule',
        schedule_type_limit='Dimensionless')

    zone_needing_cooling_actuator = openstudio_model.EnergyManagementSystemActuator(
        sch_zones_needing_cooling, 'Schedule:Year', 'Schedule Value')
    zone_needing_cooling_actuator.setName('Zones_Needing_Cooling')

    # create container schedules to hold ratio of zones needing heating and cooling
    sch_zones_needing_heating_ratio = create_constant_schedule_ruleset(
        model, 0, name='Zones Needing Heating Ratio Schedule',
        schedule_type_limit='Dimensionless')

    zone_needing_heating_ratio_actuator = openstudio_model.EnergyManagementSystemActuator(
        sch_zones_needing_heating_ratio, 'Schedule:Year', 'Schedule Value')
    zone_needing_heating_ratio_actuator.setName('Zone_Heating_Ratio')

    sch_zones_needing_cooling_ratio = create_constant_schedule_ruleset(
        model, 0, name='Zones Needing Cooling Ratio Schedule',
        schedule_type_limit='Dimensionless')

    zone_needing_cooling_ratio_actuator = openstudio_model.EnergyManagementSystemActuator(
        sch_zones_needing_cooling_ratio, 'Schedule:Year', 'Schedule Value')
    zone_needing_cooling_ratio_actuator.setName('Zone_Cooling_Ratio')

    #####
    # Create EMS program to check comfort exceedances
    ####

    # initalize inner body for heating and cooling requests programs
    determine_zone_cooling_needs_prg_inner_body = ''
    determine_zone_heating_needs_prg_inner_body = ''

    for zone in thermal_zones:
        # get existing 'sensors'
        exisiting_ems_sensors = model.getEnergyManagementSystemSensors()
        exisiting_ems_sensors_names = []
        for sen in exisiting_ems_sensors:
            sen_desc = '{}-{}'.format(sen.nameString(), sen.outputVariableOrMeterName())
            exisiting_ems_sensors_names.append(sen_desc)

        # Create zone air temperature 'sensor' for the zone.
        zone_name = ems_friendly_name(zone.nameString())
        zone_air_sensor_name = '{}_ctrl_temperature'.format(zone_name)

        if '{}-Zone Air Temperature'.format(zone_air_sensor_name) \
                not in exisiting_ems_sensors_names:
            zone_ctrl_temperature = openstudio_model.EnergyManagementSystemSensor(
                model, 'Zone Air Temperature')
            zone_ctrl_temperature.setName(zone_air_sensor_name)
            zone_ctrl_temperature.setKeyName(zone.nameString())

        # check for zone thermostats
        zone_thermostat = zone.thermostatSetpointDualSetpoint()
        if not zone_thermostat.is_initialized():
            raise ValueError('Zone {} does not have thermostats.'.format(zone.nameString()))

        zone_thermostat = zone.thermostatSetpointDualSetpoint().get()
        zone_clg_thermostat = zone_thermostat.coolingSetpointTemperatureSchedule().get()
        zone_htg_thermostat = zone_thermostat.heatingSetpointTemperatureSchedule.get()

        # create new sensor for zone thermostat if it does not exist already
        zone_clg_thermostat_sensor_name = '{}_upper_comfort_limit'.format(zone_name)
        zone_htg_thermostat_sensor_name = '{}_lower_comfort_limit'.format(zone_name)

        if '{}-Schedule Value'.format(zone_clg_thermostat_sensor_name) \
                not in exisiting_ems_sensors_names:
            # Upper comfort limit for the zone. Taken from existing thermostat
            zone_upper_comfort_limit = openstudio_model.EnergyManagementSystemSensor(
                model, 'Schedule Value')
            zone_upper_comfort_limit.setName(zone_clg_thermostat_sensor_name)
            zone_upper_comfort_limit.setKeyName(zone_clg_thermostat.nameString())

        if '{}-Schedule Value'.format(zone_htg_thermostat_sensor_name) \
                not in exisiting_ems_sensors_names:
            # Lower comfort limit for the zone. Taken from existing thermostat schedules in the zone.
            zone_lower_comfort_limit = openstudio_model.EnergyManagementSystemSensor(
                model, 'Schedule Value')
            zone_lower_comfort_limit.setName(zone_htg_thermostat_sensor_name)
            zone_lower_comfort_limit.setKeyName(zone_htg_thermostat.nameString())

        # create program inner body for determining zone cooling needs
        z_cool_need = \
            'IF {zone_air_sensor_name} > {zone_clg_thermostat_sensor_name},\n' \
            'SET Zones_Needing_Cooling = Zones_Needing_Cooling + 1,\n' \
            'ENDIF,\n'.format(
                zone_air_sensor_name=zone_air_sensor_name,
                zone_clg_thermostat_sensor_name=zone_clg_thermostat_sensor_name
            )
        determine_zone_cooling_needs_prg_inner_body += z_cool_need

        # create program inner body for determining zone cooling needs
        z_heat_need = \
            'IF {zone_air_sensor_name} < {zone_htg_thermostat_sensor_name},\n' \
            'SET Zones_Needing_Heating = Zones_Needing_Heating + 1,\n' \
            'ENDIF,\n'.format(
                zone_air_sensor_name=zone_air_sensor_name,
                zone_htg_thermostat_sensor_name=zone_htg_thermostat_sensor_name
            )
        determine_zone_heating_needs_prg_inner_body += z_heat_need

    # create program for determining zone cooling needs
    determine_zone_cooling_needs_prg = \
        openstudio_model.EnergyManagementSystemProgram(model)
    determine_zone_cooling_needs_prg.setName('Determine_Zone_Cooling_Needs')
    determine_zone_cooling_needs_prg_body = \
        'SET Zones_Needing_Cooling = 0,\n' \
        '{zone_cooling_prg_inner_body}' \
        'SET Total_Zones = {thermal_zones_length}\n,' \
        'SET Zone_Cooling_Ratio = Zones_Needing_Cooling/Total_Zones'.format(
            zone_cooling_prg_inner_body=determine_zone_cooling_needs_prg_inner_body,
            thermal_zones_length=len(thermal_zones)
        )
    determine_zone_cooling_needs_prg.setBody(determine_zone_cooling_needs_prg_body)

    # create program for determining zone heating needs
    determine_zone_heating_needs_prg = \
        openstudio_model.EnergyManagementSystemProgram(model)
    determine_zone_heating_needs_prg.setName('Determine_Zone_Heating_Needs')
    determine_zone_heating_needs_prg_body = \
        'SET Zones_Needing_Heating = 0,\n' \
        '{zone_heating_prg_inner_body}\n' \
        'SET Total_Zones = {thermal_zones_length},\n' \
        'SET Zone_Heating_Ratio = Zones_Needing_Heating/Total_Zones'.format(
            zone_heating_prg_inner_body=determine_zone_heating_needs_prg_inner_body,
            thermal_zones_length=len(thermal_zones)
        )
    determine_zone_heating_needs_prg.setBody(determine_zone_heating_needs_prg_body)

    # create EMS program manager objects
    programs_at_beginning_of_timestep = \
        openstudio_model.EnergyManagementSystemProgramCallingManager(model)
    programs_at_beginning_of_timestep.setName(
        'Heating_Cooling_Request_Programs_At_End_Of_Timestep')
    programs_at_beginning_of_timestep.setCallingPoint(
        'EndOfZoneTimestepAfterZoneReporting')
    programs_at_beginning_of_timestep.addProgram(determine_zone_cooling_needs_prg)
    programs_at_beginning_of_timestep.addProgram(determine_zone_heating_needs_prg)


def model_add_plant_supply_water_temperature_control(
        model, plant_water_loop, control_strategy='outdoor_air',
        sp_at_oat_low=None, oat_low=None, sp_at_oat_high=None, oat_high=None,
        thermal_zones=()):
    """Adds supply water temperature control on specified plant water loops.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        plant_water_loop: [OpenStudio::Model::PlantLoop] plant water loop to add
            supply water temperature control.
        control_strategy: [String] Method to determine how to control the plant's
            supply water temperature (swt). Options include the following.
            outdoor_air -- The plant's swt will be proportional to the outdoor
                air based on the next 4 parameters.
            zone_demand -- The plant's swt will be determined by preponderance
                of zone demand.
        sp_at_oat_low: [Double] supply water temperature setpoint, in F, at the
            outdoor low temperature.
        oat_low: [Double] outdoor drybulb air  temperature, in F, for low setpoint.
        sp_at_oat_high: [Double] supply water temperature setpoint, in F, at
            the outdoor high temperature.
        oat_high: [Double] outdoor drybulb air temperature, in F, for high setpoint.
        thermal_zones: [Array<OpenStudio::Model::ThermalZone>] array of zones.
    """
    loop_name = plant_water_loop.nameString()
    # check that all required temperature parameters are defined
    if all(val is None for val in (sp_at_oat_low, oat_low, sp_at_oat_high, oat_high)):
        print('At least one of the required temperature parameter is nil.')

    # remove any existing setpoint manager on the plant water loop
    exisiting_setpoint_managers = plant_water_loop.loopTemperatureSetpointNode().setpointManagers()
    for spm in exisiting_setpoint_managers:
        spm.disconnect()

    if control_strategy == 'outdoor_air':
        # create supply water temperature setpoint managers for plant based on outdoor temperature
        water_loop_setpoint_manager = openstudio_model.SetpointManagerOutdoorAirReset(model)
        water_loop_setpoint_manager.setName(
            '{} Supply Water Temperature Control'.format(loop_name))
        water_loop_setpoint_manager.setControlVariable('Temperature')
        water_loop_setpoint_manager.setSetpointatOutdoorLowTemperature(
            TEMPERATURE.to_unit([sp_at_oat_low], 'C', 'F')[0])
        water_loop_setpoint_manager.setOutdoorLowTemperature(
            TEMPERATURE.to_unit([oat_low], 'C', 'F')[0])
        water_loop_setpoint_manager.setSetpointatOutdoorHighTemperature(
            TEMPERATURE.to_unit([sp_at_oat_high], 'C', 'F')[0])
        water_loop_setpoint_manager.setOutdoorHighTemperature(
            TEMPERATURE.to_unit([oat_high], 'C', 'F')[0])
        water_loop_setpoint_manager.addToNode(
            plant_water_loop.loopTemperatureSetpointNode())
    else:
        # create supply water temperature setpoint managers for plant based on
        # zone heating and cooling demand
        # check if zone heat and cool requests program exists, if not create it
        determine_zone_cooling_needs_prg = \
            model.getEnergyManagementSystemProgramByName('Determine_Zone_Cooling_Needs')
        determine_zone_heating_needs_prg = \
            model.getEnergyManagementSystemProgramByName('Determine_Zone_Heating_Needs')
        if not determine_zone_cooling_needs_prg.is_initialized() and not \
                determine_zone_heating_needs_prg.is_initialized():
            model_add_zone_heat_cool_request_count_program(model, thermal_zones)

        plant_water_loop_name = ems_friendly_name(loop_name)

        if plant_water_loop.componentType().valueName() == 'Heating':
            sp_at_oat_low = 120 if sp_at_oat_low is None else sp_at_oat_low
            swt_upper_limit = TEMPERATURE.to_unit([sp_at_oat_low], 'C', 'F')[0]
            sp_at_oat_high = 80 if sp_at_oat_high is None else sp_at_oat_high
            swt_lower_limit = TEMPERATURE.to_unit([sp_at_oat_high], 'C', 'F')[0]
            swt_init = TEMPERATURE.to_unit([100], 'C', 'F')[0]
            zone_demand_var = 'Zone_Heating_Ratio'
            swt_inc_condition_var = '> 0.70'
            swt_dec_condition_var = '< 0.30'
        else:
            sp_at_oat_low = 70 if sp_at_oat_low is None else sp_at_oat_low
            swt_upper_limit = TEMPERATURE.to_unit([sp_at_oat_low], 'C', 'F')[0]
            sp_at_oat_high = 55 if sp_at_oat_high is None else sp_at_oat_high
            swt_lower_limit = TEMPERATURE.to_unit([sp_at_oat_high], 'C', 'F')[0]
            swt_init = TEMPERATURE.to_unit([62], 'C', 'F')[0]
            zone_demand_var = 'Zone_Cooling_Ratio'
            swt_inc_condition_var = '< 0.30'
            swt_dec_condition_var = '> 0.70'

        # plant loop supply water control actuator
        sch_plant_swt_ctrl = create_constant_schedule_ruleset(
            model, swt_init,
            name='{}_Sch_Supply_Water_Temperature'.format(plant_water_loop_name),
            schedule_type_limit='Temperature')

        cmd_plant_water_ctrl = openstudio_model.EnergyManagementSystemActuator(
            sch_plant_swt_ctrl, 'Schedule:Year', 'Schedule Value')
        cmd_plant_water_ctrl.setName('{}_supply_water_ctrl'.format(plant_water_loop_name))

        # create plant loop setpoint manager
        water_loop_setpoint_manager = \
            openstudio_model.SetpointManagerScheduled(model, sch_plant_swt_ctrl)
        water_loop_setpoint_manager.setName(
            '{} Supply Water Temperature Control'.format(loop_name))
        water_loop_setpoint_manager.setControlVariable('Temperature')
        water_loop_setpoint_manager.addToNode(
            plant_water_loop.loopTemperatureSetpointNode())

        # add uninitialized variables into constant program
        set_constant_values_prg_body = \
            'SET {}_supply_water_ctrl = {}'.format(plant_water_loop_name, swt_init)

        set_constant_values_prg = model.getEnergyManagementSystemProgramByName(
            'Set_Plant_Constant_Values')
        if set_constant_values_prg.is_initialized():
            set_constant_values_prg = set_constant_values_prg.get()
            set_constant_values_prg.addLine(set_constant_values_prg_body)
        else:
            set_constant_values_prg = \
                openstudio_model.EnergyManagementSystemProgram(model)
            set_constant_values_prg.setName('Set_Plant_Constant_Values')
            set_constant_values_prg.setBody(set_constant_values_prg_body)

        # program for supply water temperature control in the plot
        determine_plant_swt_prg = openstudio_model.EnergyManagementSystemProgram(model)
        swt_prg_name = 'Determine_{}_Supply_Water_Temperature'.format(plant_water_loop_name)
        determine_plant_swt_prg.setName(swt_prg_name)
        determine_plant_swt_prg_body = \
            'SET SWT_Increase = 1,\n' \
            'SET SWT_Decrease = 1,\n' \
            'SET SWT_upper_limit = {swt_upper_limit},\n' \
            'SET SWT_lower_limit = {swt_lower_limit},\n' \
            'IF {zone_demand_var} {swt_inc_cond_var} && (@Mod CurrentTime 1) == 0,\n' \
            'SET {loop_name}_supply_water_ctrl = {loop_name}_supply_water_ctrl + SWT_Increase,\n' \
            'ELSEIF {zone_demand_var} {swt_dec_cond_var} && (@Mod CurrentTime 1) == 0,\n' \
            'SET {loop_name}_supply_water_ctrl = {loop_name}_supply_water_ctrl - SWT_Decrease,\n' \
            'ELSE,\n' \
            'SET {loop_name}_supply_water_ctrl = {loop_name}_supply_water_ctrl,\n' \
            'ENDIF,\n' \
            'IF {loop_name}_supply_water_ctrl > SWT_upper_limit,\n' \
            'SET {loop_name}_supply_water_ctrl = SWT_upper_limit\n' \
            'ENDIF,\n' \
            'IF {loop_name}_supply_water_ctrl < SWT_lower_limit,\n' \
            'SET {loop_name}_supply_water_ctrl = SWT_lower_limit\n' \
            'ENDIF'.format(
                swt_upper_limit=swt_upper_limit, swt_lower_limit=swt_lower_limit,
                zone_demand_var=zone_demand_var, loop_name=plant_water_loop_name,
                swt_inc_cond_var=swt_inc_condition_var, swt_dec_cond_var=swt_dec_condition_var
            )
        determine_plant_swt_prg.setBody(determine_plant_swt_prg_body)

        # create EMS program manager objects
        programs_at_beginning_of_timestep = \
            openstudio_model.EnergyManagementSystemProgramCallingManager(model)
        prg_man_name = '{}_Demand_Based_Supply_Water_Temperature_At_Beginning_'\
            'Of_Timestep'.format(plant_water_loop_name)
        programs_at_beginning_of_timestep.setName(prg_man_name)
        programs_at_beginning_of_timestep.setCallingPoint('BeginTimestepBeforePredictor')
        programs_at_beginning_of_timestep.addProgram(determine_plant_swt_prg)

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


def model_get_or_add_chilled_water_loop(
        model, cool_fuel, chilled_water_loop_cooling_type='WaterCooled'):
    """Get existing chilled water loop or add a new one if there isn't one already.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        cool_fuel: [String] the cooling fuel. Valid choices are Electricity,
            DistrictCooling, and HeatPump.
        chilled_water_loop_cooling_type: [String] Archetype for chilled water
            loops, AirCooled or WaterCooled.
    """
    # retrieve the existing chilled water loop or add a new one if necessary
    chilled_water_loop = None
    if model.getPlantLoopByName('Chilled Water Loop').is_initialized():
        chilled_water_loop = model.getPlantLoopByName('Chilled Water Loop').get()
    else:
        if cool_fuel == 'DistrictCooling':
            chilled_water_loop = model_add_chw_loop(
                model, chw_pumping_type='const_pri', cooling_fuel=cool_fuel)
        elif cool_fuel == 'HeatPump':
            condenser_water_loop = model_get_or_add_ambient_water_loop(model)
            chilled_water_loop = model_add_chw_loop(
                model, chw_pumping_type='const_pri_var_sec',
                chiller_cooling_type='WaterCooled',
                chiller_compressor_type='Rotary Screw',
                condenser_water_loop=condenser_water_loop)
        elif cool_fuel == 'Electricity':
            if chilled_water_loop_cooling_type == 'AirCooled':
                chilled_water_loop = model_add_chw_loop(
                    model, chw_pumping_type='const_pri',
                    chiller_cooling_type='AirCooled', cooling_fuel=cool_fuel)
            else:
                fan_type = 'Variable Speed Fan'
                condenser_water_loop = model_add_cw_loop(
                    model, cooling_tower_type='Open Cooling Tower',
                    cooling_tower_fan_type='Propeller or Axial',
                    cooling_tower_capacity_control=fan_type,
                    number_of_cells_per_tower=1, number_cooling_towers=1)
                chilled_water_loop = model_add_chw_loop(
                    model, chw_pumping_type='const_pri_var_sec',
                    chiller_cooling_type='WaterCooled',
                    chiller_compressor_type='Rotary Screw',
                    condenser_water_loop=condenser_water_loop)
        else:
            print('No cool_fuel specified.')

    return chilled_water_loop


def model_get_or_add_hot_water_loop(
        model, heat_fuel, hot_water_loop_type='HighTemperature'):
    """Get existing hot water loop or add a new one if there isn't one already.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        heat_fuel: [String] the heating fuel. Valid choices are NaturalGas,
            Electricity, DistrictHeating, DistrictHeatingWater, DistrictHeatingSteam.
        hot_water_loop_type: [String] Archetype for hot water loops.
    """
    if heat_fuel is None:
        raise ValueError('Hot water loop fuel type is None. Cannot add hot water loop.')

    make_new_hot_water_loop = True
    hot_water_loop = None
    # retrieve the existing hot water loop or add a new one if not of the correct type
    if model.getPlantLoopByName('Hot Water Loop').is_initialized():
        hot_water_loop = model.getPlantLoopByName('Hot Water Loop').get()
        design_loop_exit_temperature = \
            hot_water_loop.sizingPlant().designLoopExitTemperature()
        design_loop_exit_temperature = \
            TEMPERATURE.to_unit([design_loop_exit_temperature], 'F', 'C')[0]
        # check that the loop is the correct archetype
        if hot_water_loop_type == 'HighTemperature':
            if design_loop_exit_temperature > 130.0:
                make_new_hot_water_loop = False
        elif hot_water_loop_type == 'LowTemperature':
            if design_loop_exit_temperature <= 130.0:
                make_new_hot_water_loop = False

    if make_new_hot_water_loop:
        if hot_water_loop_type == 'HighTemperature':
            hot_water_loop = model_add_hw_loop(model, heat_fuel)
        elif hot_water_loop_type == 'LowTemperature':
            hot_water_loop = model_add_hw_loop(
                model, heat_fuel, dsgn_sup_wtr_temp=120.0,
                boiler_draft_type='Condensing')
        else:
            msg = 'Hot water loop archetype {} not recognized.'.format(hot_water_loop_type)
            raise ValueError(msg)
    return hot_water_loop


def model_get_or_add_ambient_water_loop(model):
    """Get the existing ambient water loop or add a new one if there isn't one already.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
    """
    # retrieve the existing hot water loop or add a new one if necessary
    exist_loop = model.getPlantLoopByName('Ambient Loop')
    if exist_loop.is_initialized():
        ambient_water_loop = exist_loop.get()
    else:
        ambient_water_loop = model_add_district_ambient_loop(model)
    return ambient_water_loop


def model_get_or_add_ground_hx_loop(model):
    """Get the existing ground heat exchanger loop or add a new one if there isn't one.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
    """
    # retrieve the existing ground HX loop or add a new one if necessary
    exist_loop = model.getPlantLoopByName('Ground HX Loop')
    if exist_loop.is_initialized():
        ground_hx_loop = exist_loop.get()
    else:
        ground_hx_loop = model_add_ground_hx_loop(model)
    return ground_hx_loop


def model_get_or_add_heat_pump_loop(
        model, heat_fuel, cool_fuel,
        heat_pump_loop_cooling_type='EvaporativeFluidCooler'):
    # retrieve the existing heat pump loop or add a new one if necessary
    exist_loop = model.getPlantLoopByName('Heat Pump Loop')
    if exist_loop.is_initialized():
        heat_pump_loop = exist_loop.get()
    else:
        heat_pump_loop = model_add_hp_loop(
            model, heating_fuel=heat_fuel, cooling_fuel=cool_fuel,
            cooling_type=heat_pump_loop_cooling_type)
    return heat_pump_loop


def model_add_hvac_system(
        model, system_type, main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
        hot_water_loop_type='HighTemperature',
        chilled_water_loop_cooling_type='WaterCooled',
        heat_pump_loop_cooling_type='EvaporativeFluidCooler',
        air_loop_heating_type='Water', air_loop_cooling_type='Water',
        zone_equipment_ventilation=True, fan_coil_capacity_control_method='CyclingFan'):
    """Add the a system type to the zones based on the specified template.

    For multi-zone system types, add one system per story.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        system_type: [String] The system type.
        main_heat_fuel: [String] Main heating fuel used for air loops and plant loops.
        zone_heat_fuel: [String] Zone heating fuel for zone hvac equipment and
            terminal units.
        cool_fuel: [String] Cooling fuel used for air loops, plant loops, and
            zone equipment.
        zones: [Array<OpenStudio::Model::ThermalZone>] array of thermal zones
            served by the system.
        hot_water_loop_type: [String] Archetype for hot water loops. Either
            HighTemperature (180F supply) (default) or LowTemperature (120F supply).
            Only used if HVAC system has a hot water loop.
        chilled_water_loop_cooling_type [String] Archetype for chilled water loops.
            Only used if HVAC system has a chilled water loop and cool_fuel
            is Electricity. Options are.

            * AirCooled
            * WaterCooled

        heat_pump_loop_cooling_type: [String] the type of cooling equipment for
            heat pump loops if not DistrictCooling. Valid options are.

            * CoolingTower
            * CoolingTowerSingleSpeed
            * CoolingTowerTwoSpeed
            * CoolingTowerVariableSpeed
            * FluidCooler
            * FluidCoolerSingleSpeed
            * FluidCoolerTwoSpeed
            * EvaporativeFluidCooler
            * EvaporativeFluidCoolerSingleSpeed
            * EvaporativeFluidCoolerTwoSpeed

        air_loop_heating_type: [String] type of heating coil serving main air loop.
            Options are.

            * Gas
            * DX
            * Water

        air_loop_cooling_type: [String] type of cooling coil serving main air loop.
            Options are.

            * DX
            * Water

        zone_equipment_ventilation: [Boolean] toggle whether to include outdoor air
            ventilation on zone equipment including as fan coil units, VRF terminals,
            or water source heat pumps.
        fan_coil_capacity_control_method: [String] Only applicable to Fan Coil
            system type. Capacity control method for the fan coil. If VariableFan,
            the fan will be VariableVolume. Options are.

            * ConstantFanVariableFlow
            * CyclingFan
            * VariableFanVariableFlow
            * VariableFanConstantFlow.

    Returns:
        Returns True if successful, False if not.
    """
    # enforce defaults if fields are None
    if hot_water_loop_type is None:
        hot_water_loop_type = 'HighTemperature'
    if chilled_water_loop_cooling_type is None:
        chilled_water_loop_cooling_type = 'WaterCooled'
    if heat_pump_loop_cooling_type is None:
        heat_pump_loop_cooling_type = 'EvaporativeFluidCooler'
    if air_loop_heating_type is None:
        air_loop_heating_type = 'Water'
    if air_loop_cooling_type is None:
        air_loop_cooling_type = 'Water'
    if zone_equipment_ventilation is None:
        zone_equipment_ventilation = True
    if fan_coil_capacity_control_method is None:
        fan_coil_capacity_control_method = 'CyclingFan'

    # don't do anything if there are no zones
    if len(zones) == 0:
        return None

    # add the different types of systems
    air_loop = None
    if system_type == 'PTAC':
        water_types = ('NaturalGas', 'DistrictHeating',
                       'DistrictHeatingWater', 'DistrictHeatingSteam')
        if main_heat_fuel in water_types:
            heating_type = 'Water'
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'AirSourceHeatPump':
            heating_type = 'Water'
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type='LowTemperature')
        elif main_heat_fuel == 'Electricity':
            heating_type = main_heat_fuel
            hot_water_loop = None
        else:
            heating_type = zone_heat_fuel
            hot_water_loop = None

        model_add_ptac(
            model, zones, cooling_type='Single Speed DX AC', heating_type=heating_type,
            hot_water_loop=hot_water_loop, fan_type='Cycling',
            ventilation=zone_equipment_ventilation)

    elif system_type == 'PTHP':
        model_add_pthp(
            model, zones, fan_type='Cycling', ventilation=zone_equipment_ventilation)

    elif system_type == 'PSZ-AC':
        if main_heat_fuel in ('NaturalGas', 'Gas'):
            heating_type = main_heat_fuel
            supplemental_heating_type = 'Electricity'
            if air_loop_heating_type == 'Water':
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
                heating_type = 'Water'
            else:
                hot_water_loop = None
        elif main_heat_fuel in ('DistrictHeating', 'DistrictHeatingWater',
                                'DistrictHeatingSteam'):
            heating_type = 'Water'
            supplemental_heating_type = 'Electricity'
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel in ('AirSourceHeatPump', 'ASHP'):
            heating_type = 'Water'
            supplemental_heating_type = 'Electricity'
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'Electricity':
            heating_type = main_heat_fuel
            supplemental_heating_type = 'Electricity'
        else:
            heating_type = zone_heat_fuel
            supplemental_heating_type = None
            hot_water_loop = None

        if cool_fuel == 'DistrictCooling':
            chilled_water_loop = model_get_or_add_chilled_water_loop(model, cool_fuel)
            cooling_type = 'Water'
        else:
            chilled_water_loop = None
            cooling_type = 'Single Speed DX AC'

        air_loop = model_add_psz_ac(
            model, zones, cooling_type=cooling_type, chilled_water_loop=chilled_water_loop,
            hot_water_loop=hot_water_loop, heating_type=heating_type,
            supplemental_heating_type=supplemental_heating_type,
            fan_location='DrawThrough', fan_type='ConstantVolume')

    elif system_type == 'PSZ-HP':
        air_loop = model_add_psz_ac(
            model, zones, system_name='PSZ-HP', cooling_type='Single Speed Heat Pump',
            heating_type='Single Speed Heat Pump', supplemental_heating_type='Electricity',
            fan_location='DrawThrough', fan_type='ConstantVolume')

    elif system_type == 'PSZ-VAV':
        supplemental_heating_type = None if main_heat_fuel is None else 'Electricity'
        air_loop = model_add_psz_vav(
            model, zones, system_name='PSZ-VAV', heating_type=main_heat_fuel,
            supplemental_heating_type=supplemental_heating_type,
            hvac_op_sch=None, oa_damper_sch=None)

    elif system_type == 'VRF':
        model_add_vrf(model, zones, ventilation=zone_equipment_ventilation)

    elif system_type == 'Fan Coil':
        water_types = ('NaturalGas', 'DistrictHeating', 'DistrictHeatingWater',
                       'DistrictHeatingSteam', 'Electricity')
        if main_heat_fuel in water_types:
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'AirSourceHeatPump':
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type='LowTemperature')
        else:
            hot_water_loop = None

        if cool_fuel in ('Electricity', 'DistrictCooling'):
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        model_add_four_pipe_fan_coil(
            model, zones, chilled_water_loop, hot_water_loop=hot_water_loop,
            ventilation=zone_equipment_ventilation,
            capacity_control_method=fan_coil_capacity_control_method)

    elif system_type == 'Radiant Slab':
        water_types = ('NaturalGas', 'DistrictHeating', 'DistrictHeatingWater',
                       'DistrictHeatingSteam', 'Electricity')
        if main_heat_fuel in water_types:
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'AirSourceHeatPump':
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type='LowTemperature')
        else:
            hot_water_loop = None

        if cool_fuel in ('Electricity', 'DistrictCooling'):
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        model_add_low_temp_radiant(model, zones, hot_water_loop, chilled_water_loop)

    elif system_type == 'Baseboards':
        water_types = ('NaturalGas', 'DistrictHeating', 'DistrictHeatingWater',
                       'DistrictHeatingSteam')
        if main_heat_fuel in water_types:
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'AirSourceHeatPump':
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type='LowTemperature')
        elif main_heat_fuel == 'Electricity':
            hot_water_loop = None
        else:
            raise ValueError('Baseboards must have heating_type specified.')
        model_add_baseboard(model, zones, hot_water_loop=hot_water_loop)

    elif system_type == 'Unit Heaters':
        model_add_unitheater(
            model, zones, hvac_op_sch=None, fan_control_type='ConstantVolume',
            fan_pressure_rise=0.2, heating_type=main_heat_fuel)

    elif system_type == 'High Temp Radiant':
        model_add_high_temp_radiant(
            model, zones, heating_type=main_heat_fuel, combustion_efficiency=0.8)

    elif system_type == 'Window AC':
        model_add_window_ac(model, zones)

    elif system_type == 'Residential AC':
        model_add_furnace_central_ac(
            model, zones, heating=False, cooling=True, ventilation=False)

    elif system_type == 'Forced Air Furnace':
        model_add_furnace_central_ac(
            model, zones, heating=True, cooling=False, ventilation=True,
            heating_type=main_heat_fuel)

    elif system_type == 'Residential Forced Air Furnace':
        model_add_furnace_central_ac(
            model, zones, heating=True, cooling=False, ventilation=False)

    elif system_type == 'Residential Forced Air Furnace with AC':
        model_add_furnace_central_ac(
            model, zones, heating=True, cooling=True, ventilation=False)

    elif system_type == 'Residential Air Source Heat Pump':
        heating = False if main_heat_fuel is None else True
        cooling = False if cool_fuel is None else True
        model_add_central_air_source_heat_pump(
            model, zones, heating=heating, cooling=cooling, ventilation=False)

    elif system_type == 'Residential Minisplit Heat Pumps':
        model_add_minisplit_hp(model, zones)

    elif system_type == 'VAV Reheat':
        water_types = ('NaturalGas', 'Gas', 'HeatPump', 'DistrictHeating',
                       'DistrictHeatingWater', 'DistrictHeatingSteam')
        if main_heat_fuel in water_types:
            heating_type = main_heat_fuel
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'AirSourceHeatPump':
            heating_type = main_heat_fuel
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type='LowTemperature')
        else:
            heating_type = 'Electricity'
            hot_water_loop = None

        if air_loop_cooling_type == 'Water':
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        if hot_water_loop is None:
            if zone_heat_fuel in ('NaturalGas', 'Gas'):
                reheat_type = 'NaturalGas'
            elif zone_heat_fuel == 'Electricity':
                reheat_type = 'Electricity'
            else:
                msg = 'zone_heat_fuel "{}" is not supported with main_heat_fuel "{}" ' \
                    'for a "VAV Reheat" system type.'.format(
                       zone_heat_fuel, main_heat_fuel)
                raise ValueError(msg)
        else:
            reheat_type = 'Water'

        air_loop = model_add_vav_reheat(
            model, zones, heating_type=heating_type, reheat_type=reheat_type,
            hot_water_loop=hot_water_loop, chilled_water_loop=chilled_water_loop,
            fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0)

    elif system_type == 'VAV No Reheat':
        water_types = ('NaturalGas', 'Gas', 'HeatPump', 'DistrictHeating',
                       'DistrictHeatingWater', 'DistrictHeatingSteam')
        if main_heat_fuel in water_types:
            heating_type = main_heat_fuel
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        elif main_heat_fuel == 'AirSourceHeatPump':
            heating_type = main_heat_fuel
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type='LowTemperature')
        else:
            heating_type = 'Electricity'
            hot_water_loop = None

        if air_loop_cooling_type == 'Water':
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        air_loop = model_add_vav_reheat(
            model, zones, heating_type=heating_type, reheat_type=None,
            hot_water_loop=hot_water_loop, chilled_water_loop=chilled_water_loop,
            fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0)

    elif system_type == 'VAV Gas Reheat':
        if air_loop_cooling_type == 'Water':
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        air_loop = model_add_vav_reheat(
            model, zones, heating_type='NaturalGas', reheat_type='NaturalGas',
            chilled_water_loop=chilled_water_loop, fan_efficiency=0.62,
            fan_motor_efficiency=0.9, fan_pressure_rise=4.0)

    elif system_type == 'PVAV Reheat':
        if main_heat_fuel == 'AirSourceHeatPump':
            hot_water_loop = model_get_or_add_hot_water_loop(
                model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        else:
            if air_loop_heating_type == 'Water':
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
            else:
                heating_type = main_heat_fuel

        if cool_fuel == 'Electricity':
            chilled_water_loop = None
        else:
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)

        electric_reheat = True if zone_heat_fuel == 'Electricity' else False

        air_loop = model_add_pvav(
            model, zones,
            hot_water_loop=hot_water_loop, chilled_water_loop=chilled_water_loop,
            heating_type=main_heat_fuel, electric_reheat=electric_reheat)

    elif system_type == 'PVAV PFP Boxes':
        if cool_fuel == 'DistrictCooling':
            chilled_water_loop = model_get_or_add_chilled_water_loop(model, cool_fuel)
        else:
            chilled_water_loop = None
        air_loop = model_add_pvav_pfp_boxes(
            model, zones, chilled_water_loop=chilled_water_loop,
            fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0)

    elif system_type == 'VAV PFP Boxes':
        chilled_water_loop = model_get_or_add_chilled_water_loop(
            model, cool_fuel,
            chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        air_loop = model_add_vav_pfp_boxes(
            model, zones, chilled_water_loop=chilled_water_loop,
            fan_efficiency=0.62, fan_motor_efficiency=0.9, fan_pressure_rise=4.0)

    elif system_type == 'Water Source Heat Pumps':
        if ('DistrictHeating' in main_heat_fuel and cool_fuel == 'DistrictCooling') or \
                (main_heat_fuel == 'AmbientLoop' and cool_fuel == 'AmbientLoop'):
            condenser_loop = model_get_or_add_ambient_water_loop(model)
        else:
            condenser_loop = model_get_or_add_heat_pump_loop(
                model, main_heat_fuel, cool_fuel,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type)

        model_add_water_source_hp(
            model, zones, condenser_loop, ventilation=zone_equipment_ventilation)

    elif system_type == 'Ground Source Heat Pumps':
        condenser_loop = model_get_or_add_ground_hx_loop(model)
        model_add_water_source_hp(
            model, zones, condenser_loop, ventilation=zone_equipment_ventilation)

    elif system_type == 'DOAS Cold Supply':
        hot_water_loop = model_get_or_add_hot_water_loop(
            model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        chilled_water_loop = model_get_or_add_chilled_water_loop(
            model, cool_fuel,
            chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        air_loop = model_add_doas_cold_supply(
            model, zones,
            hot_water_loop=hot_water_loop, chilled_water_loop=chilled_water_loop)

    elif system_type == 'DOAS':
        if air_loop_heating_type == 'Water':
            if main_heat_fuel is None:
                hot_water_loop = None
            elif main_heat_fuel == 'AirSourceHeatPump':
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type='LowTemperature')
            elif main_heat_fuel == 'Electricity':
                msg = 'air_loop_heating_type "{}" is not supported with main_heat_fuel ' \
                    '"{}" for a "DOAS" system type.'.format(
                        air_loop_heating_type, main_heat_fuel)
                raise ValueError(msg)
            else:
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        else:
            hot_water_loop = None

        if air_loop_cooling_type == 'Water':
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        air_loop = model_add_doas(
            model, zones, hot_water_loop=hot_water_loop,
            chilled_water_loop=chilled_water_loop)

    elif system_type == 'DOAS with DCV':
        if air_loop_heating_type == 'Water':
            if main_heat_fuel is None:
                hot_water_loop = None
            elif main_heat_fuel == 'AirSourceHeatPump':
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type='LowTemperature')
            else:
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        else:
            hot_water_loop = None

        if air_loop_cooling_type == 'Water':
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        air_loop = model_add_doas(
            model, zones,
            hot_water_loop=hot_water_loop, chilled_water_loop=chilled_water_loop,
            doas_type='DOASVAV', demand_control_ventilation=True)

    elif system_type == 'DOAS with Economizing':
        if air_loop_heating_type == 'Water':
            if main_heat_fuel is None:
                hot_water_loop = None
            elif main_heat_fuel == 'AirSourceHeatPump':
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type='LowTemperature')
            else:
                hot_water_loop = model_get_or_add_hot_water_loop(
                    model, main_heat_fuel, hot_water_loop_type=hot_water_loop_type)
        else:
            hot_water_loop = None

        if air_loop_cooling_type == 'Water':
            chilled_water_loop = model_get_or_add_chilled_water_loop(
                model, cool_fuel,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        else:
            chilled_water_loop = None

        air_loop = model_add_doas(
            model, zones,
            hot_water_loop=hot_water_loop, chilled_water_loop=chilled_water_loop,
            doas_type='DOASVAV', econo_ctrl_mthd='FixedDryBulb')

    elif system_type == 'ERVs':
        model_add_zone_erv(model, zones)

    elif system_type == 'Residential ERVs':
        model_add_residential_erv(model, zones)

    elif system_type == 'Residential Ventilators':
        model_add_residential_ventilator(model, zones)

    elif system_type == 'Evaporative Cooler':
        model_add_evap_cooler(model, zones)

    else:  # Combination Systems
        if 'with DOAS with DCV' in system_type:
            # add DOAS DCV system
            air_loop = model_add_hvac_system(
                model, 'DOAS with DCV', main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
                hot_water_loop_type=hot_water_loop_type,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type,
                air_loop_heating_type=air_loop_heating_type,
                air_loop_cooling_type=air_loop_cooling_type,
                zone_equipment_ventilation=False,
                fan_coil_capacity_control_method=fan_coil_capacity_control_method)
            # add paired system type
            paired_system_type = system_type.replace(' with DOAS with DCV', '')
            model_add_hvac_system(
                model, paired_system_type, main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
                hot_water_loop_type=hot_water_loop_type,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type,
                air_loop_heating_type=air_loop_heating_type,
                air_loop_cooling_type=air_loop_cooling_type,
                zone_equipment_ventilation=False,
                fan_coil_capacity_control_method=fan_coil_capacity_control_method)
        elif 'with DOAS' in system_type:
            # add DOAS system
            air_loop = model_add_hvac_system(
                model, 'DOAS', main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
                hot_water_loop_type=hot_water_loop_type,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type,
                air_loop_heating_type=air_loop_heating_type,
                air_loop_cooling_type=air_loop_cooling_type,
                zone_equipment_ventilation=False,
                fan_coil_capacity_control_method=fan_coil_capacity_control_method)
            # add paired system type
            paired_system_type = system_type.replace(' with DOAS', '')
            model_add_hvac_system(
                model, paired_system_type, main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
                hot_water_loop_type=hot_water_loop_type,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type,
                air_loop_heating_type=air_loop_heating_type,
                air_loop_cooling_type=air_loop_cooling_type,
                zone_equipment_ventilation=False,
                fan_coil_capacity_control_method=fan_coil_capacity_control_method)
        elif 'with ERVs' in system_type:
            # add DOAS system
            model_add_hvac_system(
                model, 'ERVs', main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
                hot_water_loop_type=hot_water_loop_type,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type,
                air_loop_heating_type=air_loop_heating_type,
                air_loop_cooling_type=air_loop_cooling_type,
                zone_equipment_ventilation=False,
                fan_coil_capacity_control_method=fan_coil_capacity_control_method)
            # add paired system type
            paired_system_type = system_type.replace(' with ERVs', '')
            model_add_hvac_system(
                model, paired_system_type, main_heat_fuel, zone_heat_fuel, cool_fuel, zones,
                hot_water_loop_type=hot_water_loop_type,
                chilled_water_loop_cooling_type=chilled_water_loop_cooling_type,
                heat_pump_loop_cooling_type=heat_pump_loop_cooling_type,
                air_loop_heating_type=air_loop_heating_type,
                air_loop_cooling_type=air_loop_cooling_type,
                zone_equipment_ventilation=False,
                fan_coil_capacity_control_method=fan_coil_capacity_control_method)
        else:
            raise ValueError('HVAC system type "{}" not recognized'.format(system_type))

    return air_loop
