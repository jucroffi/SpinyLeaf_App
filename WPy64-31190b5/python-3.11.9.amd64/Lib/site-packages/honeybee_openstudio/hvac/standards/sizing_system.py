# coding=utf-8
"""Modules taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.CoilCoolingWater.rb
"""
from __future__ import division

from honeybee_openstudio.openstudio import openstudio


def adjust_sizing_system(air_loop_hvac, dsgn_temps, type_of_load_sizing='Sensible',
                         min_sys_airflow_ratio=0.3, sizing_option='Coincident'):
    """Prototype SizingSystem object

    Args:
        air_loop_hvac: [OpenStudio::Model::AirLoopHVAC] air loop.
        dsgn_temps: [Hash] a hash of design temperature lookups from
            standard_design_sizing_temperatures.
    """
    # adjust sizing system defaults
    sizing_system = air_loop_hvac.sizingSystem()
    sizing_system.setTypeofLoadtoSizeOn(type_of_load_sizing)
    sizing_system.autosizeDesignOutdoorAirFlowRate()
    sizing_system.setPreheatDesignTemperature(dsgn_temps['prehtg_dsgn_sup_air_temp_c'])
    sizing_system.setPrecoolDesignTemperature(dsgn_temps['preclg_dsgn_sup_air_temp_c'])
    sizing_system.setCentralCoolingDesignSupplyAirTemperature(
       dsgn_temps['clg_dsgn_sup_air_temp_c'])
    sizing_system.setCentralHeatingDesignSupplyAirTemperature(
       dsgn_temps['htg_dsgn_sup_air_temp_c'])
    sizing_system.setPreheatDesignHumidityRatio(0.008)
    sizing_system.setPrecoolDesignHumidityRatio(0.008)
    sizing_system.setCentralCoolingDesignSupplyAirHumidityRatio(0.0085)
    sizing_system.setCentralHeatingDesignSupplyAirHumidityRatio(0.0080)
    if air_loop_hvac.model().version() < openstudio.VersionString('2.7.0'):
        sizing_system.setMinimumSystemAirFlowRatio(min_sys_airflow_ratio)
    else:
        sizing_system.setCentralHeatingMaximumSystemAirFlowRatio(min_sys_airflow_ratio)
    sizing_system.setSizingOption(sizing_option)
    sizing_system.setAllOutdoorAirinCooling(False)
    sizing_system.setAllOutdoorAirinHeating(False)
    sizing_system.setSystemOutdoorAirMethod('ZoneSum')
    sizing_system.setCoolingDesignAirFlowMethod('DesignDay')
    sizing_system.setHeatingDesignAirFlowMethod('DesignDay')

    return sizing_system
