# coding=utf-8
"""OpenStudio translators for template HVAC systems."""
from __future__ import division

from honeybee.typing import clean_ep_string
from honeybee_energy.hvac.allair._base import _AllAirBase
from honeybee_energy.hvac.doas._base import _DOASBase
from honeybee_energy.hvac.allair.ptac import PTAC
from honeybee_energy.hvac.allair.psz import PSZ
from honeybee_energy.hvac.allair.pvav import PVAV
from honeybee_energy.hvac.allair.vav import VAV
from honeybee_energy.hvac.allair.furnace import ForcedAirFurnace
from honeybee_energy.hvac.doas.fcu import FCUwithDOAS
from honeybee_energy.hvac.doas.vrf import VRFwithDOAS
from honeybee_energy.hvac.doas.wshp import WSHPwithDOAS
from honeybee_energy.hvac.doas.radiant import RadiantwithDOAS
from honeybee_energy.hvac.heatcool.baseboard import Baseboard
from honeybee_energy.hvac.heatcool.evapcool import EvaporativeCooler
from honeybee_energy.hvac.heatcool.fcu import FCU
from honeybee_energy.hvac.heatcool.gasunit import GasUnitHeater
from honeybee_energy.hvac.heatcool.residential import Residential
from honeybee_energy.hvac.heatcool.vrf import VRF
from honeybee_energy.hvac.heatcool.windowac import WindowAC
from honeybee_energy.hvac.heatcool.wshp import WSHP
from honeybee_energy.hvac.heatcool.radiant import Radiant

from honeybee_openstudio.openstudio import openstudio, openstudio_model, \
    OSScheduleRuleset, OSTime
from .standards.hvac_systems import model_add_hvac_system, model_add_low_temp_radiant, \
    model_get_or_add_chilled_water_loop, model_add_hw_loop, model_add_chw_loop, \
    model_add_cw_loop, model_get_or_add_ambient_water_loop


def template_hvac_to_openstudio(hvac, os_zones, os_model):
    """Convert Honeybee HVAC TemplateSystem to OpenStudio.

    Args:
        hvac: Any honeybee-energy TemplateSystem class instance to be translated
            to OpenStudio.
        os_zones: A dictionary with two keys, each of which has a value for a
            list of OpenStudio ThermalZones. The keys are heated_zones and
            cooled_zones and the lists under each key note the OpenStudio
            ThermalZones to be given heating and cooling equipment by the HVAC.
        os_model: The OpenStudio Model object to which the HVAC system
            will be added.
    """
    # unpack the heated and cooled zones and organize them into groups
    heated_zones, cooled_zones = os_zones['heated_zones'], os_zones['cooled_zones']
    heated_and_cooled_zones, cooled_only_zones, heated_only_zones = [], [], []
    heat_dict = {z.nameString(): z for z in heated_zones}
    cool_dict = {z.nameString(): z for z in cooled_zones}
    all_dict = heat_dict.copy()
    all_dict.update(cool_dict)
    zones = list(all_dict.values())
    for zone in zones:
        zone_name = zone.nameString()
        if zone_name in heat_dict and zone_name in cool_dict:
            heated_and_cooled_zones.append(zone)
        elif zone_name in cool_dict:
            cooled_only_zones.append(zone)
        else:
            heated_only_zones.append(zone)
    system_zones = heated_and_cooled_zones + cooled_only_zones

    # determine the DOAS type from the demand controlled ventilation
    dcv = getattr(hvac, 'demand_controlled_ventilation', False)
    doas_type = 'DOAS' if not dcv else 'DOAS with DCV'
    air_loop = None  # will be returned from HVAC creation

    # add the HVAC system equipment using the equipment type
    # system type naming convention:
    # [ventilation strategy] [cooling system and plant] [heating system and plant]
    equip = hvac.equipment_type
    if isinstance(hvac, Baseboard):
        if equip == 'ElectricBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'BoilerBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'ASHPBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_zones)

        elif equip == 'DHWBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_zones)

    elif isinstance(hvac, EvaporativeCooler):
        if equip == 'EvapCoolers_ElectricBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)
            model_add_hvac_system(os_model, 'Evaporative Cooler',
                                  None, None, 'Electricity', cooled_zones)

        elif equip == 'EvapCoolers_BoilerBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)
            model_add_hvac_system(os_model, 'Evaporative Cooler', None, None,
                                  'Electricity', cooled_zones)

        elif equip == 'EvapCoolers_ASHPBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_zones)
            model_add_hvac_system(os_model, 'Evaporative Cooler', None, None,
                                  'Electricity', cooled_zones)

        elif equip == 'EvapCoolers_DHWBaseboard':
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_zones)
            model_add_hvac_system(os_model, 'Evaporative Cooler', None, None,
                                  'Electricity', cooled_zones)

        elif equip == 'EvapCoolers_Furnace':
            # use unit heater to represent forced air furnace
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)
            model_add_hvac_system(os_model, 'Evaporative Cooler', None, None,
                                  'Electricity', cooled_zones)

        elif equip == 'EvapCoolers_UnitHeaters':
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)
            model_add_hvac_system(os_model, 'Evaporative Cooler', None, None,
                                  'Electricity', cooled_zones)

        elif equip == 'EvapCoolers':
            model_add_hvac_system(os_model, 'Evaporative Cooler', None, None,
                                  'Electricity', cooled_zones)

    elif isinstance(hvac, FCUwithDOAS):
        if equip == 'DOAS_FCU_Chiller_Boiler':
            air_loop = model_add_hvac_system(os_model, doas_type, 'NaturalGas', None,
                                             'Electricity', zones)
            model_add_hvac_system(os_model, 'Fan Coil', 'NaturalGas', None, 'Electricity',
                                  zones, zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_Chiller_ASHP':
            air_loop = model_add_hvac_system(os_model, doas_type, 'AirSourceHeatPump',
                                             None, 'Electricity', zones)
            model_add_hvac_system(os_model, 'Fan Coil', 'AirSourceHeatPump', None,
                                  'Electricity', zones, zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_Chiller_DHW':
            air_loop = model_add_hvac_system(os_model, doas_type, 'DistrictHeating',
                                             None, 'Electricity', zones)
            model_add_hvac_system(os_model, 'Fan Coil', 'DistrictHeating', None,
                                  'Electricity', zones, zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_Chiller_ElectricBaseboard':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'Electricity', zones)
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  zone_equipment_ventilation=False)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity', None,
                                  None, heated_zones)

        elif equip == 'DOAS_FCU_Chiller_GasHeaters':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'Electricity', zones)
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  zone_equipment_ventilation=False)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas', None,
                                  None, heated_zones)

        elif equip == 'DOAS_FCU_Chiller':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'Electricity', zones)
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_ACChiller_Boiler':
            air_loop = model_add_hvac_system(os_model, doas_type, 'NaturalGas', None,
                                             'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Fan Coil', 'NaturalGas', None, 'Electricity',
                                  zones, chilled_water_loop_cooling_type='AirCooled',
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_ACChiller_ASHP':
            air_loop = model_add_hvac_system(os_model, doas_type, 'AirSourceHeatPump', None,
                                             'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Fan Coil', 'AirSourceHeatPump', None,
                                  'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled',
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_ACChiller_DHW':
            air_loop = model_add_hvac_system(os_model, doas_type, 'DistrictHeating', None,
                                             'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Fan Coil', 'DistrictHeating', None,
                                  'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled',
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_ACChiller_ElectricBaseboard':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled',
                                  zone_equipment_ventilation=False)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'DOAS_FCU_ACChiller_GasHeaters':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled',
                                  zone_equipment_ventilation=False)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas', None,
                                  None, heated_zones)

        elif equip == 'DOAS_FCU_ACChiller':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled',
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_DCW_Boiler':
            air_loop = model_add_hvac_system(os_model, doas_type, 'NaturalGas', None,
                                             'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Fan Coil', 'NaturalGas', None,
                                  'DistrictCooling', zones,
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_DCW_ASHP':
            air_loop = model_add_hvac_system(os_model, doas_type, 'AirSourceHeatPump',
                                             None, 'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Fan Coil', 'AirSourceHeatPump',
                                  None, 'DistrictCooling', zones,
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_DCW_DHW':
            air_loop = model_add_hvac_system(os_model, doas_type, 'DistrictHeating',
                                             None, 'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Fan Coil', 'DistrictHeating',
                                  None, 'DistrictCooling', zones,
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_FCU_DCW_ElectricBaseboard':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'DistrictCooling',
                                  zones, zone_equipment_ventilation=False)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'DOAS_FCU_DCW_GasHeaters':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'DistrictCooling',
                                  zones, zone_equipment_ventilation=False)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'DOAS_FCU_DCW':
            air_loop = model_add_hvac_system(os_model, doas_type, None, None,
                                             'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'DistrictCooling',
                                  zones, zone_equipment_ventilation=False)

    elif isinstance(hvac, RadiantwithDOAS):
        chilled_water_loop_cooling_type = 'WaterCooled'
        if equip == 'DOAS_Radiant_Chiller_Boiler':
            main_heat_fuel, cool_fuel = 'NaturalGas', 'Electricity'
        elif equip == 'DOAS_Radiant_Chiller_ASHP':
            main_heat_fuel, cool_fuel = 'AirSourceHeatPump', 'Electricity'
        elif equip == 'DOAS_Radiant_Chiller_DHW':
            main_heat_fuel, cool_fuel = 'DistrictHeating', 'Electricity'
        elif equip == 'DOAS_Radiant_ACChiller_Boiler':
            main_heat_fuel, cool_fuel = 'NaturalGas', 'Electricity'
            chilled_water_loop_cooling_type = 'AirCooled'
        elif equip == 'DOAS_Radiant_ACChiller_ASHP':
            main_heat_fuel, cool_fuel = 'AirSourceHeatPump', 'Electricity'
            chilled_water_loop_cooling_type = 'AirCooled'
        elif equip == 'DOAS_Radiant_ACChiller_DHW':
            main_heat_fuel, cool_fuel = 'DistrictHeating', 'Electricity'
            chilled_water_loop_cooling_type = 'AirCooled'
        elif equip == 'DOAS_Radiant_DCW_Boiler':
            main_heat_fuel, cool_fuel = 'NaturalGas', 'DistrictCooling'
        elif equip == 'DOAS_Radiant_DCW_ASHP':
            main_heat_fuel, cool_fuel = 'AirSourceHeatPump', 'DistrictCooling'
        elif equip == 'DOAS_Radiant_DCW_DHW':
            main_heat_fuel, cool_fuel = 'DistrictHeating', 'DistrictCooling'

        air_loop = model_add_hvac_system(
            os_model, doas_type, main_heat_fuel, None, cool_fuel, zones,
            chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)
        hw_name = 'Low Temp Hot Water Loop'
        if os_model.getPlantLoopByName(hw_name).is_initialized():
            hot_water_loop = os_model.getPlantLoopByName(hw_name).get()
        else:
            hot_water_loop = model_add_hw_loop(
                os_model, main_heat_fuel, dsgn_sup_wtr_temp=120.0,
                boiler_draft_type='Condensing')
            hot_water_loop.setName(hw_name)
        chw_name = 'Low Temp Chilled Water Loop'
        if os_model.getPlantLoopByName(chw_name).is_initialized():
            chilled_water_loop = os_model.getPlantLoopByName(chw_name).get()
        else:
            if cool_fuel == 'DistrictCooling':
                chilled_water_loop = model_add_chw_loop(
                    os_model, chw_pumping_type='const_pri', cooling_fuel=cool_fuel)
            elif cool_fuel == 'HeatPump':
                condenser_water_loop = model_get_or_add_ambient_water_loop(os_model)
                chilled_water_loop = model_add_chw_loop(
                    os_model, chw_pumping_type='const_pri_var_sec',
                    chiller_cooling_type='WaterCooled',
                    chiller_compressor_type='Rotary Screw',
                    condenser_water_loop=condenser_water_loop)
            elif cool_fuel == 'Electricity':
                if chilled_water_loop_cooling_type == 'AirCooled':
                    chilled_water_loop = model_add_chw_loop(
                        os_model, chw_pumping_type='const_pri',
                        chiller_cooling_type='AirCooled', cooling_fuel=cool_fuel)
                else:
                    cond_name = 'Condenser Water Loop'
                    if os_model.getPlantLoopByName(cond_name).is_initialized():
                        condenser_water_loop = os_model.getPlantLoopByName(cond_name).get()
                    else:
                        fan_type = 'Variable Speed Fan'
                        condenser_water_loop = model_add_cw_loop(
                            os_model, cooling_tower_type='Open Cooling Tower',
                            cooling_tower_fan_type='Propeller or Axial',
                            cooling_tower_capacity_control=fan_type,
                            number_of_cells_per_tower=1, number_cooling_towers=1)
                        condenser_water_loop.setName(cond_name)
                    chilled_water_loop = model_add_chw_loop(
                        os_model, chw_pumping_type='const_pri_var_sec',
                        chiller_cooling_type='WaterCooled',
                        chiller_compressor_type='Rotary Screw',
                        condenser_water_loop=condenser_water_loop)
            chilled_water_loop.setName(chw_name)

        control_strategy, include_carpet = 'proportional_control', False
        radiant_temperature_control_type = 'SurfaceFaceTemperature'
        if hvac.radiant_type in ('CeilingMetalPanel', 'FloorWithHardwood'):
            control_strategy = 'none'
            radiant_temperature_control_type = 'OperativeTemperature'
        radiant_type = hvac.radiant_type.lower()
        if hvac.radiant_type == 'FloorWithCarpet':
            radiant_type, include_carpet = 'floor', True

        model_add_low_temp_radiant(
            os_model, zones, hot_water_loop, chilled_water_loop, radiant_type=radiant_type,
            include_carpet=include_carpet, control_strategy=control_strategy,
            radiant_temperature_control_type=radiant_temperature_control_type,
            radiant_availability_type='all_day')

    elif isinstance(hvac, VRFwithDOAS):
        if equip == 'DOAS_VRF':
            air_loop = model_add_hvac_system(os_model, doas_type, 'Electricity',
                                             None, 'Electricity', zones,
                                             air_loop_heating_type='DX',
                                             air_loop_cooling_type='DX')
            model_add_hvac_system(os_model, 'VRF', 'Electricity',
                                  None, 'Electricity', zones)

    elif isinstance(hvac, WSHPwithDOAS):
        if equip == 'DOAS_WSHP_FluidCooler_Boiler':
            air_loop = model_add_hvac_system(os_model, doas_type, 'NaturalGas', None,
                                             'Electricity', zones)
            model_add_hvac_system(os_model, 'Water Source Heat Pumps', 'NaturalGas',
                                  None, 'Electricity', zones,
                                  heat_pump_loop_cooling_type='FluidCooler',
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_WSHP_CoolingTower_Boiler':
            air_loop = model_add_hvac_system(os_model, doas_type, 'NaturalGas', None,
                                             'Electricity', zones)
            model_add_hvac_system(os_model, 'Water Source Heat Pumps', 'NaturalGas',
                                  None, 'Electricity', zones,
                                  heat_pump_loop_cooling_type='CoolingTower',
                                  zone_equipment_ventilation=False)

        elif equip == 'DOAS_WSHP_GSHP':
            air_loop = model_add_hvac_system(os_model, doas_type, 'Electricity', None,
                                             'Electricity', zones, air_loop_heating_type='DX',
                                             air_loop_cooling_type='DX')
            model_add_hvac_system(os_model, 'Ground Source Heat Pumps', 'Electricity', None,
                                  'Electricity', zones, zone_equipment_ventilation=False)

        elif equip == 'DOAS_WSHP_DCW_DHW':
            air_loop = model_add_hvac_system(os_model, doas_type, 'DistrictHeating', None,
                                             'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Water Source Heat Pumps', 'DistrictHeating',
                                  None, 'DistrictCooling', zones,
                                  zone_equipment_ventilation=False)

    # ventilation provided by zone fan coil unit in fan coil systems
    elif isinstance(hvac, FCU):
        if equip == 'FCU_Chiller_Boiler':
            model_add_hvac_system(os_model, 'Fan Coil', 'NaturalGas', None,
                                  'Electricity', zones)

        elif equip == 'FCU_Chiller_ASHP':
            model_add_hvac_system(os_model, 'Fan Coil', 'AirSourceHeatPump',
                                  None, 'Electricity', zones)

        elif equip == 'FCU_Chiller_DHW':
            model_add_hvac_system(os_model, 'Fan Coil', 'DistrictHeating',
                                  None, 'Electricity', zones)

        elif equip == 'FCU_Chiller_ElectricBaseboard':
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'FCU_Chiller_GasHeaters':
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'FCU_Chiller':
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones)

        elif equip == 'FCU_ACChiller_Boiler':
            model_add_hvac_system(os_model, 'Fan Coil', 'NaturalGas', None, 'Electricity',
                                  zones, chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'FCU_ACChiller_ASHP':
            model_add_hvac_system(os_model, 'Fan Coil', 'AirSourceHeatPump',
                                  None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'FCU_ACChiller_DHW':
            model_add_hvac_system(os_model, 'Fan Coil', 'DistrictHeating', None,
                                  'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'FCU_ACChiller_ElectricBaseboard':
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'FCU_ACChiller_GasHeaters':
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled')
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'FCU_ACChiller':
            model_add_hvac_system(os_model, 'Fan Coil', None, None, 'Electricity', zones,
                                  chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'FCU_DCW_Boiler':
            model_add_hvac_system(os_model, 'Fan Coil ', 'NaturalGas', None,
                                  'DistrictCooling', zones)

        elif equip == 'FCU_DCW_ASHP':
            model_add_hvac_system(os_model, 'Fan Coil', 'AirSourceHeatPump',
                                  None, 'DistrictCooling', zones)

        elif equip == 'FCU_DCW_DHW':
            model_add_hvac_system(os_model, 'Fan Coil', 'DistrictHeating',
                                  None, 'DistrictCooling', zones)

        elif equip == 'FCU_DCW_ElectricBaseboard':
            model_add_hvac_system(os_model, 'Fan Coil', None, None,
                                  'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'FCU_DCW_GasHeaters':
            model_add_hvac_system(os_model, 'Fan Coil', None, None,
                                  'DistrictCooling', zones)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'FCU_DCW':
            model_add_hvac_system(os_model, 'Fan Coil', None, None,
                                  'DistrictCooling', zones)

    elif isinstance(hvac, Radiant):
        chilled_water_loop_cooling_type = 'WaterCooled'
        if equip == 'Radiant_Chiller_Boiler':
            main_heat_fuel, cool_fuel = 'NaturalGas', 'Electricity'
        elif equip == 'Radiant_Chiller_ASHP':
            main_heat_fuel, cool_fuel = 'AirSourceHeatPump', 'Electricity'
        elif equip == 'Radiant_Chiller_DHW':
            main_heat_fuel, cool_fuel = 'DistrictHeating', 'Electricity'
        elif equip == 'Radiant_ACChiller_Boiler':
            main_heat_fuel, cool_fuel = 'NaturalGas', 'Electricity'
            chilled_water_loop_cooling_type = 'AirCooled'
        elif equip == 'Radiant_ACChiller_ASHP':
            main_heat_fuel, cool_fuel = 'AirSourceHeatPump', 'Electricity'
            chilled_water_loop_cooling_type = 'AirCooled'
        elif equip == 'Radiant_ACChiller_DHW':
            main_heat_fuel, cool_fuel = 'DistrictHeating', 'Electricity'
            chilled_water_loop_cooling_type = 'AirCooled'
        elif equip == 'Radiant_DCW_Boiler':
            main_heat_fuel, cool_fuel = 'NaturalGas', 'DistrictCooling'
        elif equip == 'Radiant_DCW_ASHP':
            main_heat_fuel, cool_fuel = 'AirSourceHeatPump', 'DistrictCooling'
        elif equip == 'Radiant_DCW_DHW':
            main_heat_fuel, cool_fuel = 'DistrictHeating', 'DistrictCooling'

        hw_name = 'Low Temperature Hot Water Loop'
        if os_model.getPlantLoopByName(hw_name).is_initialized():
            hot_water_loop = os_model.getPlantLoopByName(hw_name).get()
        else:
            hot_water_loop = model_add_hw_loop(
                os_model, main_heat_fuel, dsgn_sup_wtr_temp=120.0,
                boiler_draft_type='Condensing')
            hot_water_loop.setName(hw_name)
        chilled_water_loop = model_get_or_add_chilled_water_loop(
            os_model, cool_fuel,
            chilled_water_loop_cooling_type=chilled_water_loop_cooling_type)

        control_strategy, include_carpet = 'proportional_control', False
        radiant_temperature_control_type = 'SurfaceFaceTemperature'
        if hvac.radiant_type in ('CeilingMetalPanel', 'FloorWithHardwood'):
            control_strategy = 'none'
            radiant_temperature_control_type = 'OperativeTemperature'
        radiant_type = hvac.radiant_type.lower()
        if hvac.radiant_type == 'FloorWithCarpet':
            radiant_type, include_carpet = 'floor', True

        model_add_low_temp_radiant(
            os_model, zones, hot_water_loop, chilled_water_loop, radiant_type=radiant_type,
            include_carpet=include_carpet, control_strategy=control_strategy,
            radiant_temperature_control_type=radiant_temperature_control_type,
            radiant_availability_type='all_day')

    elif isinstance(hvac, ForcedAirFurnace):
        if equip == 'Furnace':
            # includes ventilation, whereas residential forced air furnace does not.
            model_add_hvac_system(os_model, 'Forced Air Furnace', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'Furnace_Electric':
            # includes ventilation, whereas residential forced air furnace does not.
            model_add_hvac_system(os_model, 'Forced Air Furnace', 'Electricity',
                                  None, None, heated_zones)

    elif isinstance(hvac, GasUnitHeater):
        if equip == 'GasHeaters':
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

    elif isinstance(hvac, PTAC):
        if equip == 'PTAC_ElectricBaseboard':
            model_add_hvac_system(os_model, 'PTAC', None, None, 'Electricity',
                                  system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'PTAC_BoilerBaseboard':
            model_add_hvac_system(os_model, 'PTAC', None, None, 'Electricity',
                                  system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'PTAC_DHWBaseboard':
            model_add_hvac_system(os_model, 'PTAC', None, None, 'Electricity',
                                  system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_zones)

        elif equip == 'PTAC_GasHeaters':
            model_add_hvac_system(os_model, 'PTAC', None, None, 'Electricity',
                                  system_zones)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'PTAC_ElectricCoil':
            model_add_hvac_system(os_model, 'PTAC', None, 'Electricity',
                                  'Electricity', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

        elif equip == 'PTAC_GasCoil':
            model_add_hvac_system(os_model, 'PTAC', None, 'NaturalGas',
                                  'Electricity', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

        elif equip == 'PTAC_Boiler':
            model_add_hvac_system(os_model, 'PTAC', 'NaturalGas', None,
                                  'Electricity', system_zones)
            # use 'Baseboard gas boiler' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_only_zones)

        elif equip == 'PTAC_ASHP':
            model_add_hvac_system(os_model, 'PTAC', 'AirSourceHeatPump',
                                  None, 'Electricity', system_zones)
            # use 'Baseboard central air source heat pump' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_only_zones)

        elif equip == 'PTAC_DHW':
            model_add_hvac_system(os_model, 'PTAC', 'DistrictHeating', None,
                                  'Electricity', system_zones)
            # use 'Baseboard district hot water heat' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_only_zones)

        elif equip == 'PTAC':
            model_add_hvac_system(os_model, 'PTAC', None, None, 'Electricity',
                                  system_zones)

        elif equip == 'PTHP':
            model_add_hvac_system(os_model, 'PTHP', 'Electricity', None,
                                  'Electricity', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

    elif isinstance(hvac, PSZ):
        if equip == 'PSZAC_ElectricBaseboard':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'Electricity', system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_BoilerBaseboard':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'Electricity', system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_DHWBaseboard':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'Electricity', system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_GasHeaters':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'Electricity', system_zones)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_ElectricCoil':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, 'Electricity',
                                             'Electricity', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_GasCoil':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, 'NaturalGas',
                                             'Electricity', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_Boiler':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', 'NaturalGas', None,
                                             'Electricity', system_zones)
            # use 'Baseboard gas boiler' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_ASHP':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', 'AirSourceHeatPump',
                                             None, 'Electricity', system_zones)
            # use 'Baseboard central air source heat pump' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_DHW':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', 'DistrictHeating',
                                             None, 'Electricity', system_zones)
            # use 'Baseboard district hot water' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'Electricity', cooled_zones)

        elif equip == 'PSZAC_DCW_ElectricBaseboard':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'DistrictCooling', system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_DCW_BoilerBaseboard':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'DistrictCooling', system_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_DCW_GasHeaters':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'DistrictCooling', system_zones)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'PSZAC_DCW_ElectricCoil':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, 'Electricity',
                                             'DistrictCooling', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_DCW_GasCoil':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, 'NaturalGas',
                                             'DistrictCooling', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_DCW_Boiler':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', 'NaturalGas', None,
                                             'DistrictCooling', system_zones)
            # use 'Baseboard gas boiler' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_DCW_ASHP':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', 'AirSourceHeatPump',
                                             None, 'DistrictCooling', system_zones)
            # use 'Baseboard central air source heat pump' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_DCW_DHW':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', 'DistrictHeating',
                                             None, 'DistrictCooling', system_zones)
            # use 'Baseboard district hot water' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_only_zones)

        elif equip == 'PSZAC_DCW':
            air_loop = model_add_hvac_system(os_model, 'PSZ-AC', None, None,
                                             'DistrictCooling', cooled_zones)

        elif equip == 'PSZHP':
            air_loop = model_add_hvac_system(os_model, 'PSZ-HP', 'Electricity', None,
                                             'Electricity', system_zones)
            # use 'Baseboard electric' for heated only zones
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_only_zones)

    elif isinstance(hvac, PVAV):  # PVAV systems by default use a DX coil for cooling
        if equip == 'PVAV_Boiler':
            air_loop = model_add_hvac_system(os_model, 'PVAV Reheat', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones)

        elif equip == 'PVAV_ASHP':
            air_loop = model_add_hvac_system(os_model, 'PVAV Reheat', 'AirSourceHeatPump',
                                             'AirSourceHeatPump', 'Electricity', zones)

        elif equip == 'PVAV_DHW':
            air_loop = model_add_hvac_system(os_model, 'PVAV Reheat', 'DistrictHeating',
                                             'DistrictHeating', 'Electricity', zones)

        elif equip == 'PVAV_PFP':
            air_loop = model_add_hvac_system(os_model, 'PVAV PFP Boxes', 'Electricity',
                                             'Electricity', 'Electricity', zones)

        elif equip == 'PVAV_BoilerElectricReheat':
            air_loop = model_add_hvac_system(os_model, 'PVAV Reheat', 'Gas',
                                             'Electricity', 'Electricity', zones)

    elif isinstance(hvac, Residential):  # all residential systems have no ventilation
        if equip == 'ResidentialAC_ElectricBaseboard':
            model_add_hvac_system(os_model, 'Residential AC', None, None, None,
                                  cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'ResidentialAC_BoilerBaseboard':
            model_add_hvac_system(os_model, 'Residential AC', None, None, None,
                                  cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'ResidentialAC_ASHPBaseboard':
            model_add_hvac_system(os_model, 'Residential AC', None, None, None,
                                  cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_zones)

        elif equip == 'ResidentialAC_DHWBaseboard':
            model_add_hvac_system(os_model, 'Residential AC', None, None, None,
                                  cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_zones)

        elif equip == 'ResidentialAC_ResidentialFurnace':
            model_add_hvac_system(os_model, 'Residential Forced Air Furnace with AC',
                                  None, None, None, zones)

        elif equip == 'ResidentialAC':
            model_add_hvac_system(os_model, 'Residential AC', None, None, None,
                                  cooled_zones)

        elif equip == 'ResidentialHP':
            model_add_hvac_system(os_model, 'Residential Air Source Heat Pump',
                                  'Electricity', None, 'Electricity', zones)

        elif equip == 'ResidentialHPNoCool':
            model_add_hvac_system(os_model, 'Residential Air Source Heat Pump',
                                  'Electricity', None, None, heated_zones)

        elif equip == 'ResidentialFurnace':
            model_add_hvac_system(os_model, 'Residential Forced Air Furnace',
                                  'NaturalGas', None, None, zones)

    elif isinstance(hvac, VAV):
        if equip == 'VAV_Chiller_Boiler':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones)

        elif equip == 'VAV_Chiller_ASHP':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'AirSourceHeatPump',
                                             'AirSourceHeatPump', 'Electricity', zones)

        elif equip == 'VAV_Chiller_DHW':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'DistrictHeating',
                                             'DistrictHeating', 'Electricity', zones)

        elif equip == 'VAV_Chiller_PFP':
            air_loop = model_add_hvac_system(os_model, 'VAV PFP Boxes', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones)

        elif equip == 'VAV_Chiller_GasCoil':
            air_loop = model_add_hvac_system(os_model, 'VAV Gas Reheat', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones)

        elif equip == 'VAV_ACChiller_Boiler':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'VAV_ACChiller_ASHP':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'AirSourceHeatPump',
                                             'AirSourceHeatPump', 'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'VAV_ACChiller_DHW':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'DistrictHeating',
                                             'DistrictHeating', 'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'VAV_ACChiller_PFP':
            air_loop = model_add_hvac_system(os_model, 'VAV PFP Boxes', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'VAV_ACChiller_GasCoil':
            air_loop = model_add_hvac_system(os_model, 'VAV Gas Reheat', 'NaturalGas',
                                             'NaturalGas', 'Electricity', zones,
                                             chilled_water_loop_cooling_type='AirCooled')

        elif equip == 'VAV_DCW_Boiler':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'NaturalGas',
                                             'NaturalGas', 'DistrictCooling', zones)

        elif equip == 'VAV_DCW_ASHP':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'AirSourceHeatPump',
                                             'AirSourceHeatPump', 'DistrictCooling', zones)

        elif equip == 'VAV_DCW_DHW':
            air_loop = model_add_hvac_system(os_model, 'VAV Reheat', 'DistrictHeating',
                                             'DistrictHeating', 'DistrictCooling', zones)

        elif equip == 'VAV_DCW_PFP':
            air_loop = model_add_hvac_system(os_model, 'VAV PFP Boxes', 'NaturalGas',
                                             'NaturalGas', 'DistrictCooling', zones)

        elif equip == 'VAV_DCW_GasCoil':
            air_loop = model_add_hvac_system(os_model, 'VAV Gas Reheat', 'NaturalGas',
                                             'NaturalGas', 'DistrictCooling', zones)

    elif isinstance(hvac, VRF):
        if equip == 'VRF':
            model_add_hvac_system(os_model, 'VRF', 'Electricity', None,
                                  'Electricity', zones)

    elif isinstance(hvac, WSHP):
        if equip == 'WSHP_FluidCooler_Boiler':
            model_add_hvac_system(os_model, 'Water Source Heat Pumps',
                                  'NaturalGas', None, 'Electricity', zones,
                                  heat_pump_loop_cooling_type='FluidCooler')

        elif equip == 'WSHP_CoolingTower_Boiler':
            model_add_hvac_system(os_model, 'Water Source Heat Pumps',
                                  'NaturalGas', None, 'Electricity', zones,
                                  heat_pump_loop_cooling_type='CoolingTower')

        elif equip == 'WSHP_GSHP':
            model_add_hvac_system(os_model, 'Ground Source Heat Pumps',
                                  'Electricity', None, 'Electricity', zones)

        elif equip == 'WSHP_DCW_DHW':
            model_add_hvac_system(os_model, 'Water Source Heat Pumps', 'DistrictHeating',
                                  None, 'DistrictCooling', zones)

    elif isinstance(hvac, WindowAC):
        if equip == 'WindowAC_ElectricBaseboard':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'Electricity',
                                  None, None, heated_zones)

        elif equip == 'WindowAC_BoilerBaseboard':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'WindowAC_ASHPBaseboard':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'AirSourceHeatPump',
                                  None, None, heated_zones)

        elif equip == 'WindowAC_DHWBaseboard':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)
            model_add_hvac_system(os_model, 'Baseboards', 'DistrictHeating',
                                  None, None, heated_zones)

        elif equip == 'WindowAC_Furnace':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)
            model_add_hvac_system(os_model, 'Forced Air Furnace', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'WindowAC_GasHeaters':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)
            model_add_hvac_system(os_model, 'Unit Heaters', 'NaturalGas',
                                  None, None, heated_zones)

        elif equip == 'WindowAC':
            model_add_hvac_system(os_model, 'Window AC', None, None,
                                  'Electricity', cooled_zones)

    else:
        raise ValueError('HVAC system type "{}" not recognized'.format(equip))

    # assign all of the properties associated with the air loop
    if air_loop is not None:
        # name the air loop with the name the user specified for the HVAC
        clean_hvac_name = clean_ep_string(hvac.display_name)
        if not isinstance(air_loop, list):
            air_loop.setName(clean_hvac_name)
            os_air_loops = [air_loop]
        else:
            os_air_loops = air_loop
            for i, loop in enumerate(os_air_loops):
                loop.setName('{} {}'.format(clean_hvac_name, i))

        # have an always available schedule ready to use if there are no user controls
        always_avail_name = 'Building HVAC Always Available'
        opt_sch = os_model.getScheduleByName(always_avail_name)
        if opt_sch.is_initialized():
            always_avail = opt_sch.get()
        else:
            always_avail = OSScheduleRuleset(os_model)
            always_avail.setName(always_avail_name)
            def_day_sch = always_avail.defaultDaySchedule()
            def_day_sch.addValue(OSTime(0, 24, 0, 0), 1)

        # assign the properties that are specific to All-Air systems
        if isinstance(hvac, _AllAirBase):
            for os_air_loop in os_air_loops:
                # set the loop to always be available
                os_air_loop.setAvailabilitySchedule(always_avail)
                # assign the economizer
                oasys = os_air_loop.airLoopHVACOutdoorAirSystem()
                if oasys.is_initialized():
                    os_oasys = oasys.get()
                    oactrl = os_oasys.getControllerOutdoorAir()
                    oactrl.setEconomizerControlType(hvac.economizer_type)
                    # assign demand controlled ventilation
                    if hvac.demand_controlled_ventilation:
                        vent_ctrl = oactrl.controllerMechanicalVentilation()
                        vent_ctrl.setDemandControlledVentilationNoFail(True)
                        oactrl.resetMinimumFractionofOutdoorAirSchedule()

        # assign the properties that are specific to DOAS systems
        if isinstance(hvac, _DOASBase):
            avail_sch = None
            if hvac.doas_availability_schedule is not None:
                sch_id = hvac.doas_availability_schedule.identifier
                schedule = os_model.getScheduleByName(sch_id)
                if schedule.is_initialized():
                    avail_sch = schedule.get()
            avail_sch = always_avail if avail_sch is None else avail_sch
            for os_air_loop in os_air_loops:
                os_air_loop.setAvailabilitySchedule(avail_sch)

        # set the heat recovery if it is specified
        if hvac.sensible_heat_recovery != 0 or hvac.latent_heat_recovery != 0:
            for os_air_loop in os_air_loops:
                heat_ex = _get_or_add_heat_recovery(os_model, os_air_loop)
                # ratio of max to standard efficiency from OpenStudio Standards
                eff_sens = hvac.sensible_heat_recovery
                heat_ex.setSensibleEffectivenessat100CoolingAirFlow(eff_sens)
                heat_ex.setSensibleEffectivenessat100HeatingAirFlow(eff_sens)
                eff_lat = hvac.latent_heat_recovery
                heat_ex.setLatentEffectivenessat100CoolingAirFlow(eff_lat)
                heat_ex.setLatentEffectivenessat100HeatingAirFlow(eff_lat)
                if os_model.version() < openstudio.VersionString('3.8.0'):
                    heat_ex.setSensibleEffectivenessat75CoolingAirFlow(eff_sens)
                    heat_ex.setSensibleEffectivenessat75HeatingAirFlow(eff_sens)
                    heat_ex.setLatentEffectivenessat75CoolingAirFlow(eff_lat)
                    heat_ex.setLatentEffectivenessat75HeatingAirFlow(eff_lat)

        # assign electric humidifier if there's an air loop and zones have humidistat
        humidistat_exists = False
        for zone in zones:
            h_stat = zone.zoneControlHumidistat()
            if h_stat.is_initialized():
                humidistat_exists = True
                if isinstance(hvac, _DOASBase):
                    z_sizing = zone.sizingZone()
                    z_sizing.setDedicatedOutdoorAirSystemControlStrategy(
                        'NeutralDehumidifiedSupplyAir')
        if humidistat_exists:
            for os_air_loop in os_air_loops:
                _add_humidifier(os_model, os_air_loop)

        # set the outdoor air controller to respect room-level ventilation schedules
        oa_sch, oa_sch_name, = None, None
        for i, zone in enumerate(zones):
            oa_spec = zone.spaces()[0].designSpecificationOutdoorAir()
            if oa_spec.is_initialized():
                oa_spec = oa_spec.get()
                space_oa_sch = oa_spec.outdoorAirFlowRateFractionSchedule()
                if space_oa_sch.is_initialized():
                    space_oa_sch = space_oa_sch.get()
                    space_oa_sch_name = space_oa_sch.nameString()
                    if i == 0 or space_oa_sch_name == oa_sch_name:
                        oa_sch, oa_sch_name = space_oa_sch, space_oa_sch_name
                    else:  # different schedules across zones; just use constant max
                        oa_sch = None
        oa_sch = always_avail if oa_sch is None else oa_sch
        for os_air_loop in os_air_loops:
            oasys = os_air_loop.airLoopHVACOutdoorAirSystem()
            if oasys.is_initialized():
                os_oasys = oasys.get()
                oactrl = os_oasys.getControllerOutdoorAir()
                oactrl.resetMinimumFractionofOutdoorAirSchedule()
                oactrl.setMinimumOutdoorAirSchedule(oa_sch)

    # if the systems are PTAC and there is ventilation, ensure the system includes it
    if isinstance(hvac, PTAC):
        always_on = os_model.alwaysOnDiscreteSchedule()
        for zone in zones:
            # check if the space type has ventilation assigned to it
            out_air = zone.spaces()[0].designSpecificationOutdoorAir()
            if out_air.is_initialized():
                # get any ventilation schedules
                vent_sched = always_on
                out_air = out_air.get()
                air_sch = out_air.outdoorAirFlowRateFractionSchedule()
                if air_sch.is_initialized():
                    vent_sched = air_sch.get()
                # get the PTAC object
                ptac = None
                for equip in zone.equipment():
                    e_name = equip.nameString()
                    if 'PTAC' in e_name:
                        ptac = os_model.getZoneHVACPackagedTerminalAirConditioner(
                            equip.handle())
                    elif 'PTHP' in e_name:
                        ptac = os_model.getZoneHVACPackagedTerminalHeatPump(
                            equip.handle())
                # assign the schedule to the PTAC object
                if ptac is not None and ptac.is_initialized():
                    ptac = ptac.get()
                    ptac.setSupplyAirFanOperatingModeSchedule(vent_sched)


def _get_or_add_heat_recovery(os_model, os_air_loop):
    """Get an existing heat exchanger in an air loop or add one if it does not exist."""
    # get an existing heat energy recovery unit from an air loop
    for supply_comp in os_air_loop.oaComponents():
        if supply_comp.to_HeatExchangerAirToAirSensibleAndLatent().is_initialized():
            return supply_comp.to_HeatExchangerAirToAirSensibleAndLatent().get()

    # create a heat recovery unit with default zero efficiencies
    heat_ex = openstudio_model.HeatExchangerAirToAirSensibleAndLatent(os_model)
    heat_ex.setEconomizerLockout(False)
    heat_ex.setName('{}_Heat Recovery Unit'.format(os_air_loop.nameString()))

    # add the heat exchanger to the air loop
    outdoor_node = os_air_loop.reliefAirNode()
    if outdoor_node.is_initialized():
        os_outdoor_node = outdoor_node.get()
        heat_ex.addToNode(os_outdoor_node)
    return heat_ex


def _add_humidifier(os_model, os_air_loop):
    """Add a humidifier to an air loop so it can meet humidification setpoints."""
    # create an electric humidifier
    humidifier = openstudio_model.HumidifierSteamElectric(os_model)
    humidifier.setName('{}_Humidifier Unit'.format(os_air_loop.nameString()))
    humid_control = openstudio_model.SetpointManagerMultiZoneHumidityMinimum(os_model)
    humid_control.setName('{}_Humidifier Controller'.format(os_air_loop.nameString()))

    # add the humidifier to the air loop
    supply_node = os_air_loop.supplyOutletNode()
    humidifier.addToNode(supply_node)
    humid_control.addToNode(supply_node)
    return humidifier
