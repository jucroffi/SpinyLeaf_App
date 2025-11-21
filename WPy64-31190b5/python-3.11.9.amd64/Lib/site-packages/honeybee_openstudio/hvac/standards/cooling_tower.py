# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.CoolingTower.rb
"""
from __future__ import division

from ladybug.datatype.temperature import Temperature
from ladybug.datatype.temperaturedelta import TemperatureDelta

from honeybee_openstudio.openstudio import openstudio_model

TEMPERATURE = Temperature()
TEMP_DELTA = TemperatureDelta()


def prototype_apply_condenser_water_temperatures(condenser_loop, design_wet_bulb_c=None):
    """Apply approach temperature sizing criteria to a condenser water loop.

    Args:
        condenser_loop: [<OpenStudio::Model::PlantLoop>] a condenser loop served
            by a cooling tower.
        design_wet_bulb_c: [Double] the outdoor design wetbulb conditions in
            degrees Celsius
    """
    sizing_plant = condenser_loop.sizingPlant()
    loop_type = sizing_plant.loopType()
    if loop_type != 'Condenser':
        return None

    # if values are absent, use the CTI rating condition 78F
    if design_wet_bulb_c is None:
        design_wet_bulb_c = TEMPERATURE.to_unit([78.0], 'C', 'F')[0]

    # EnergyPlus has a minimum limit of 68F and maximum limit of 80F for cooling towers
    design_wet_bulb_f = TEMPERATURE.to_unit([78.0], 'F', 'C')[0]
    eplus_min_design_wet_bulb_f = 68.0
    eplus_max_design_wet_bulb_f = 80.0
    if design_wet_bulb_f < eplus_min_design_wet_bulb_f:
        design_wet_bulb_f = eplus_min_design_wet_bulb_f
    elif design_wet_bulb_f > eplus_max_design_wet_bulb_f:
        design_wet_bulb_f = eplus_max_design_wet_bulb_f
    design_wet_bulb_c = TEMPERATURE.to_unit([design_wet_bulb_f], 'C', 'F')[0]

    # Determine the design CW temperature, approach, and range
    leaving_cw_t_c, approach_k, range_k = \
        prototype_condenser_water_temperatures(design_wet_bulb_c)
    approach_r = TEMP_DELTA.to_unit([10.0], 'dF', 'dC')[0]

    # Set Cooling Tower sizing parameters.
    # Only the variable speed cooling tower in E+ allows you to set the design temperatures.
    #
    # Per the documentation
    # for CoolingTowerSingleSpeed and CoolingTowerTwoSpeed
    # E+ uses the following values during sizing:
    # 95F entering water temp
    # 95F OATdb
    # 78F OATwb
    # range = loop design delta-T aka range (specified above)
    for sc in condenser_loop.supplyComponents():
        if sc.to_CoolingTowerVariableSpeed().is_initialized():
            ct = sc.to_CoolingTowerVariableSpeed().get()
            ct.setDesignInletAirWetBulbTemperature(design_wet_bulb_c)
            ct.setDesignApproachTemperature(approach_k)
            ct.setDesignRangeTemperature(range_k)

    # Set the CW sizing parameters
    # EnergyPlus autosizing routine assumes 85F and 10F temperature difference
    energyplus_design_loop_exit_temperature_c = TEMPERATURE.to_unit([85.0], 'C', 'F')[0]
    sizing_plant.setDesignLoopExitTemperature(energyplus_design_loop_exit_temperature_c)
    sizing_plant.setLoopDesignTemperatureDifference(TEMP_DELTA.to_unit([10.0], 'dC', 'dF')[0])

    # Cooling Tower operational controls
    # G3.1.3.11 - Tower shall be controlled to maintain a 70F LCnWT where weather permits,
    # floating up to leaving water at design conditions.
    float_down_to_c = TEMPERATURE.to_unit([70.0], 'C', 'F')[0]

    # get or create a setpoint manager
    cw_t_stpt_manager = None
    for spm in condenser_loop.supplyOutletNode().setpointManagers():
        if spm.to_SetpointManagerFollowOutdoorAirTemperature().is_initialized() and \
                'Setpoint Manager Follow OATwb' in spm.nameString():
            cw_t_stpt_manager = spm.to_SetpointManagerFollowOutdoorAirTemperature().get()

    if cw_t_stpt_manager is None:
        cw_t_stpt_manager = openstudio_model.SetpointManagerFollowOutdoorAirTemperature(
            condenser_loop.model)
        cw_t_stpt_manager.addToNode(condenser_loop.supplyOutletNode())

    st_pt_name = '{} Setpoint Manager Follow OATwb with {}F Approach'.format(
        condenser_loop.nameString(), round(approach_r, 1))
    cw_t_stpt_manager.setName(st_pt_name)
    cw_t_stpt_manager.setReferenceTemperatureType('OutdoorAirWetBulb')
    # At low design OATwb, it is possible to calculate
    # a maximum temperature below the minimum. In this case,
    # make the maximum and minimum the same.
    if leaving_cw_t_c < float_down_to_c:
        leaving_cw_t_c = float_down_to_c

    cw_t_stpt_manager.setMaximumSetpointTemperature(leaving_cw_t_c)
    cw_t_stpt_manager.setMinimumSetpointTemperature(float_down_to_c)
    cw_t_stpt_manager.setOffsetTemperatureDifference(approach_k)

    return True


def prototype_condenser_water_temperatures(design_oat_wb_c):
    """Determine the performance rating method specified design condenser properties.

    Water temperature, approach, and range.

    Args:
        design_oat_wb_c: [Double] the design OA wetbulb temperature (C).
    """
    design_oat_wb_f = TEMPERATURE.to_unit([design_oat_wb_c], 'F', 'C')[0]

    # 90.1-2010 G3.1.3.11 - CW supply temp = 85F or 10F approaching design wet bulb temp.
    # Design range = 10F
    # Design Temperature rise of 10F => Range: 10F
    range_r = 10.0

    # Determine the leaving CW temp
    max_leaving_cw_t_f = 85.0
    leaving_cw_t_10f_approach_f = design_oat_wb_f + 10.0
    leaving_cw_t_f = min([max_leaving_cw_t_f, leaving_cw_t_10f_approach_f])

    # Calculate the approach
    approach_r = leaving_cw_t_f - design_oat_wb_f

    # Convert to SI units
    leaving_cw_t_c = TEMPERATURE.to_unit([leaving_cw_t_f], 'C', 'F')[0]
    approach_k = TEMP_DELTA.to_unit([approach_r], 'dC', 'dF')[0]
    range_k = TEMP_DELTA.to_unit([range_r], 'dC', 'dF')[0]

    return leaving_cw_t_c, approach_k, range_k
