# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.BoilerHotWater.rb
"""
from __future__ import division

from ladybug.datatype.temperature import Temperature

from honeybee_openstudio.openstudio import openstudio, openstudio_model

TEMPERATURE = Temperature()


def create_boiler_hot_water(
        model, hot_water_loop=None, name='Boiler', fuel_type='NaturalGas',
        draft_type='Natural', nominal_thermal_efficiency=0.80,
        eff_curve_temp_eval_var='LeavingBoiler', flow_mode='LeavingSetpointModulated',
        lvg_temp_dsgn_f=180.0, out_temp_lmt_f=203.0,
        min_plr=0.0, max_plr=1.2, opt_plr=1.0, sizing_factor=None):
    """Prototype BoilerHotWater object.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        hot_water_loop: [<OpenStudio::Model::PlantLoop>] a hot water loop
            served by the boiler.
        name: [String] the name of the boiler, or nil in which case it will be defaulted.
        fuel_type: [String] type of fuel serving the boiler.
        draft_type: [String] Boiler type Condensing, MechanicalNoncondensing,
            Natural (default).
        nominal_thermal_efficiency: [Double] boiler nominal thermal efficiency.
        eff_curve_temp_eval_var: [String] LeavingBoiler or EnteringBoiler temperature
            for the boiler efficiency curve.
        flow_mode: [String] boiler flow mode.
        lvg_temp_dsgn_f: [Double] boiler leaving design temperature in degrees Fahrenheit
            note that this field is deprecated in OS versions 3.0+.
        out_temp_lmt_f: [Double] boiler outlet temperature limit in degrees Fahrenheit.
        min_plr: [Double] boiler minimum part load ratio.
        max_plr: [Double] boiler maximum part load ratio.
        opt_plr: [Double] boiler optimum part load ratio.
        sizing_factor: [Double] boiler oversizing factor.
    """
    # create the boiler
    boiler = openstudio_model.BoilerHotWater(model)
    if name is None:
        if hot_water_loop is None:
            boiler.setName('Boiler')
        else:
            boiler.setName('{} Boiler'.format(hot_water_loop.nameString()))
    else:
        boiler.setName(name)

    if fuel_type is None or fuel_type == 'Gas':
        boiler.setFuelType('NaturalGas')
    elif fuel_type == 'Propane' or fuel_type == 'PropaneGas':
        boiler.setFuelType('Propane')
    else:
        boiler.setFuelType(fuel_type)

    if nominal_thermal_efficiency is None:
        boiler.setNominalThermalEfficiency(0.8)
    else:
        boiler.setNominalThermalEfficiency(nominal_thermal_efficiency)

    if eff_curve_temp_eval_var is None:
        boiler.setEfficiencyCurveTemperatureEvaluationVariable('LeavingBoiler')
    else:
        boiler.setEfficiencyCurveTemperatureEvaluationVariable(eff_curve_temp_eval_var)

    if flow_mode is None:
        boiler.setBoilerFlowMode('LeavingSetpointModulated')
    else:
        boiler.setBoilerFlowMode(flow_mode)

    if model.version() < openstudio.VersionString('3.0.0'):
        lvg_temp_dsgn_f = 180.0 if lvg_temp_dsgn_f is None else lvg_temp_dsgn_f
        lvg_temp_dsgn_c = TEMPERATURE.to_unit([lvg_temp_dsgn_f], 'C', 'F')[0]
        boiler.setDesignWaterOutletTemperature(lvg_temp_dsgn_c)

    out_temp_lmt_f = 203.0 if out_temp_lmt_f is None else out_temp_lmt_f
    out_temp_lmt_c = TEMPERATURE.to_unit([out_temp_lmt_f], 'C', 'F')[0]
    boiler.setWaterOutletUpperTemperatureLimit(out_temp_lmt_c)

    # logic to set different defaults for condensing boilers if not specified
    if draft_type == 'Condensing':
        if model.version() < openstudio.VersionString('3.0.0') and lvg_temp_dsgn_f is None:
            # default to 120 degrees Fahrenheit (48.49 degrees Celsius)
            dw_ot = TEMPERATURE.to_unit([120.0], 'C', 'F')[0]
            boiler.setDesignWaterOutletTemperature(dw_ot)
        if nominal_thermal_efficiency is None:
            boiler.setNominalThermalEfficiency(0.96)

    min_plr = 0 if min_plr is None else min_plr
    boiler.setMinimumPartLoadRatio(min_plr)
    max_plr = 1.2 if max_plr is None else max_plr
    boiler.setMaximumPartLoadRatio(max_plr)
    opt_plr = 1.0 if opt_plr is None else opt_plr
    boiler.setOptimumPartLoadRatio(opt_plr)
    if sizing_factor is not None:
        boiler.setSizingFactor(sizing_factor)

    # add to supply side of hot water loop if specified
    if hot_water_loop is not None:
        hot_water_loop.addSupplyBranchForComponent(boiler)
    return boiler
