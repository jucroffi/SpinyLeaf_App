# coding=utf-8
"""OpenStudio SHWSystem translators."""
from __future__ import division

from honeybee_openstudio.load import _create_constant_schedule
from honeybee_openstudio.openstudio import OSPlantLoop, OSSetpointManagerScheduled, \
    OSPumpConstantSpeed, OSWaterHeaterMixed, OSCoilWaterHeatingAirToWaterHeatPump, \
    OSFanOnOff, OSWaterHeaterHeatPump

ELECTRIC_HEATERS = (
    'Electric_WaterHeater', 'HeatPump_WaterHeater', 'Electric_TanklessHeater')
FUEL_HEATERS = ()
PUMP_HEAD = 29891  # default from OpenStudio App (Pa)
PUMP_EFFICIENCY = 0.9  # default from OpenStudio App
HEAT_PUMP_DEADBAND = 4  # a default for Heat Pumps taken from the OpenStudio App (dc)
MIN_LOOP_TEMP = 10  # default from OpenStudio App (C)
LOOP_TEMP_DIFFERENCE = 5  # default from OpenStudio App (dC)


def _retrieve_source_zone(room_id, os_model):
    """Get the ThermalZone given the room_id for where a SHWSystem is placed."""
    source_zone_ref = os_model.getThermalZoneByName(room_id)
    if source_zone_ref.is_initialized():
        return source_zone_ref.get()
    else:  # the id probably points to a Space instead of a ThermalZone
        source_space_ref = os_model.getSpaceByName(room_id)
        if source_space_ref.is_initialized():
            source_space = source_space_ref.get()
            source_zone_ref = source_space.thermalZone()
            if source_zone_ref.is_initialized():
                return source_zone_ref.get()


def shw_system_to_openstudio(shw, os_shw_connections, total_flow, water_temp, os_model):
    """Translate a Honeybee SHWSystem to OpenStudio PlantLoop with equipment.

    Args:
        shw: The Honeybee-energy SHWSystem object to be translated to OpenStudio.
            This input can also be None, in which case a default system will be
            created using a plant loop with a generic District Heating water heater.
        os_shw_connections: A list of OpenStudio WaterUseConnections objects for
            all of the connections to be made to the system. These are typically
            obtained by using the hot_water_to_openstudio function in this package
            on the Room.properties.energy.service_hot_water object.
        total_flow: A number for the total flow rate of water in the system in m3/s.
            This is typically obtained by summing the individual Room flow rates
            across the system.
        water_temp: A number for the temperature of the water in Celsius. This is
            typically obtained by taking the maximum hot water temperature across
            the individual Room target_temperature.
        os_model: The OpenStudio Model to which the service hot water system
            will be added.
    """
    # set the overall properties if the system is None (default)
    if shw is None:
        shw_id = equip_type = 'Default_District_SHW'
    else:
        shw_id = shw.identifier
        equip_type = shw.equipment_type

    # create the plant loop
    hot_water_plant = OSPlantLoop(os_model)
    hot_water_plant.setName('SHW Loop {}'.format(shw_id))
    target_temp = round(water_temp, 3)
    hot_water_plant.setMaximumLoopTemperature(target_temp)
    hot_water_plant.setMinimumLoopTemperature(MIN_LOOP_TEMP)
    # edit the sizing information to be for a hot water loop
    loop_sizing = hot_water_plant.sizingPlant()
    loop_sizing.setLoopType('Heating')
    loop_sizing.setDesignLoopExitTemperature(target_temp)  
    loop_sizing.setLoopDesignTemperatureDifference(LOOP_TEMP_DIFFERENCE)
    # add a setpoint manager for the loop
    hot_sch_name = '{}C Hot Water'.format(target_temp)
    hot_sch = _create_constant_schedule(hot_sch_name, target_temp, os_model)
    sp_manager = OSSetpointManagerScheduled(os_model, hot_sch)
    sp_manager.addToNode(hot_water_plant.supplyOutletNode())
    # add a constant speed pump for the loop
    hot_water_pump = OSPumpConstantSpeed(os_model)
    hot_water_pump.setName('SHW Pump')
    hot_water_pump.setRatedPumpHead(PUMP_HEAD)
    hot_water_pump.setMotorEfficiency(PUMP_EFFICIENCY)
    hot_water_pump.addToNode(hot_water_plant.supplyInletNode())

    # add the equipment to the plant loop depending on the equipment type
    if equip_type == 'Default_District_SHW':
        # add a district heating system to supply the heat for the loop
        district_hw = OSWaterHeaterMixed(os_model)
        district_hw.setName('Ideal Service Hot Water Heater')
        district_hw.setHeaterFuelType('DistrictHeating')
        district_hw.setOffCycleParasiticFuelType('DistrictHeating')
        district_hw.setOnCycleParasiticFuelType('DistrictHeating')
        district_hw.setHeaterThermalEfficiency(1.0)
        district_hw.setHeaterMaximumCapacity(1000000)
        district_hw.setTankVolume(0)
        district_hw.setHeaterControlType('Modulate')
        a_sch_id = '22C Ambient Condition'
        a_sch = _create_constant_schedule(a_sch_id, 22, os_model)
        district_hw.setAmbientTemperatureSchedule(a_sch)
        district_hw.setOffCycleLossCoefficienttoAmbientTemperature(0)
        district_hw.setOnCycleLossCoefficienttoAmbientTemperature(0)
        hot_water_plant.addSupplyBranchForComponent(district_hw)
        # try to minimize the impact of the pump as much as possible
        hot_water_pump.setEndUseSubcategory('Water Systems')
        hot_water_pump.setMotorEfficiency(0.9)
    else:
        # add a water heater to supply the heat for the loop
        heater = OSWaterHeaterMixed(os_model)
        if equip_type in ELECTRIC_HEATERS:
            heater.setHeaterFuelType('Electricity')
            heater.setOffCycleParasiticFuelType('Electricity')
            heater.setOnCycleParasiticFuelType('Electricity')
        # set the water heater efficiency
        if equip_type == 'HeatPump_WaterHeater':
            heater.setHeaterThermalEfficiency(1.0)
        else:
            heater.setHeaterThermalEfficiency(shw.heater_efficiency)
        # set the ambient condition of the water tank
        if isinstance(shw.ambient_condition, str):  # id of a room where heater is
            heater_in_zone = True
            source_zone = _retrieve_source_zone(shw.ambient_condition, os_model)
            if source_zone is not None:
                heater.setAmbientTemperatureThermalZone(source_zone)
            heater.setAmbientTemperatureIndicator('ThermalZone')
        else:  # a temperature condition in which the heater exists
            heater_in_zone = False
            a_sch_id = '{}C Ambient Condition'.format(shw.ambient_condition)
            a_sch = _create_constant_schedule(a_sch_id, shw.ambient_condition, os_model)
            heater.setAmbientTemperatureSchedule(a_sch)
        # set the ambient loss coefficient
        if heater_in_zone:
            heater.setOffCycleLossFractiontoThermalZone(1)
            heater.setOnCycleLossFractiontoThermalZone(1)
        else:
            heater.setOffCycleLossCoefficienttoAmbientTemperature(
                shw.ambient_loss_coefficient)
            heater.setOnCycleLossCoefficienttoAmbientTemperature(
                shw.ambient_loss_coefficient)
        # set the capacity and and controls of the water heater
        heater.setHeaterMaximumCapacity(1000000)
        if equip_type in ('Gas_TanklessHeater', 'Electric_TanklessHeater'):
            heater.setName('SHW Tankless WaterHeater')
            heater.setTankVolume(0)
            heater.setHeaterControlType('Modulate')
            heater.setOffCycleLossCoefficienttoAmbientTemperature(0)
            heater.setOnCycleLossCoefficienttoAmbientTemperature(0)
        else:
            heater.setName('SHW WaterHeater')
            heater.setTankVolume(total_flow)

        # if it's a heat pump system, then add the pump
        if equip_type == 'HeatPump_WaterHeater':
            # create a coil for the heat pump
            heat_pump = OSCoilWaterHeatingAirToWaterHeatPump(os_model)
            heat_pump.setName('SHW HPWH DX Coil')
            heat_pump.setRatedCOP(shw.heater_efficiency)
            # add a fan for the heat pump system
            fan = OSFanOnOff(os_model)
            fan.setName('HPWH Fan')
            fan.setEndUseSubcategory('Water Systems')
            set_p_sch_id = 'HPWH Setpoint - {}'.format(shw_id)
            set_p_sch_val = water_temp + (HEAT_PUMP_DEADBAND * 2)
            setpt_sch = _create_constant_schedule(set_p_sch_id, set_p_sch_val, os_model)
            inlet_sch_id = 'Inlet Air Mixer Fraction - {}'.format(shw_id)
            inlet_sch = _create_constant_schedule(inlet_sch_id, 0.2, os_model)
            # add a water heater to supply the heat for the loop
            heat_sys = OSWaterHeaterHeatPump(os_model, heat_pump, heater, fan, setpt_sch, inlet_sch)
            heat_sys.setDeadBandTemperatureDifference(HEAT_PUMP_DEADBAND)
            source_zone = _retrieve_source_zone(shw.ambient_condition, os_model)
            if source_zone is not None:
                heat_sys.addToThermalZone(source_zone)
            heat_sys.setName('SHW WaterHeater HeatPump')

        # add the water heater to the loop
        hot_water_plant.addSupplyBranchForComponent(heater)

    # add all of the water use connections to the loop
    for shw_conn in os_shw_connections:
        hot_water_plant.addDemandBranchForComponent(shw_conn)
    return hot_water_plant
