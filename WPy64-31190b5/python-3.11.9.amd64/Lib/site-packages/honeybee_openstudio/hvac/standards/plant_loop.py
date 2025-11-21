# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/standards/Standards.PlantLoop.rb
"""
from __future__ import division

from ladybug.datatype.temperature import Temperature
from ladybug.datatype.temperaturedelta import TemperatureDelta

from honeybee_openstudio.openstudio import openstudio_model
from .schedule import create_constant_schedule_ruleset

TEMPERATURE = Temperature()
TEMP_DELTA = TemperatureDelta()


def chw_sizing_control(model, chilled_water_loop, dsgn_sup_wtr_temp, dsgn_sup_wtr_temp_delt):
    """Apply sizing and controls to chilled water loop.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        chilled_water_loop: [OpenStudio::Model::PlantLoop] chilled water loop.
        dsgn_sup_wtr_temp: [Double] design chilled water supply T.
        dsgn_sup_wtr_temp_delt: [Double] design chilled water supply delta T.
    """
    # chilled water loop sizing and controls
    dsgn_sup_wtr_temp = 44.0 if dsgn_sup_wtr_temp is None else dsgn_sup_wtr_temp
    dsgn_sup_wtr_temp_c = TEMPERATURE.to_unit([dsgn_sup_wtr_temp], 'C', 'F')[0]
    dsgn_sup_wtr_temp_delt = 10.1 if dsgn_sup_wtr_temp_delt is None \
        else dsgn_sup_wtr_temp_delt
    dsgn_sup_wtr_temp_delt_k = TEMP_DELTA.to_unit([dsgn_sup_wtr_temp_delt], 'dC', 'dF')[0]

    chilled_water_loop.setMinimumLoopTemperature(1.0)
    chilled_water_loop.setMaximumLoopTemperature(40.0)
    sizing_plant = chilled_water_loop.sizingPlant()
    sizing_plant.setLoopType('Cooling')
    sizing_plant.setDesignLoopExitTemperature(dsgn_sup_wtr_temp_c)
    sizing_plant.setLoopDesignTemperatureDifference(dsgn_sup_wtr_temp_delt_k)
    chw_temp_sch = create_constant_schedule_ruleset(
        model, dsgn_sup_wtr_temp_c,
        name='{} Temp - {}F'.format(chilled_water_loop.nameString(), int(dsgn_sup_wtr_temp)),
        schedule_type_limit='Temperature')
    chw_stpt_manager = openstudio_model.SetpointManagerScheduled(model, chw_temp_sch)
    chw_stpt_manager.setName('{} Setpoint Manager'.format(chilled_water_loop.nameString()))
    chw_stpt_manager.addToNode(chilled_water_loop.supplyOutletNode())
    return True


def plant_loop_set_chw_pri_sec_configuration(model):
    """Set configuration in model for chilled water primary/secondary loop interface."""
    return 'common_pipe'
