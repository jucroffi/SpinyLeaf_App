# coding=utf-8
"""OpenStudio IdealLoadsAirSystem translator."""
from __future__ import division

from honeybee.altnumber import no_limit
from honeybee.typing import clean_ep_string
from honeybee_energy.altnumber import autosize
from honeybee_energy.hvac.idealair import IdealAirSystem

from honeybee_openstudio.openstudio import OSZoneHVACIdealLoadsAirSystem


def ideal_air_system_to_openstudio(hvac, os_model, room=None):
    """Convert Honeybee IdealAirSystem to OpenStudio ZoneHVACIdealLoadsAirSystem.

    Args:
        hvac: A Honeybee-energy IdealAirSystem to be translated to OpenStudio.
        os_model: The OpenStudio Model object to which the IdealAirSystem
            will be added.
        room: An optional Honeybee Room to be used to set various properties
            of the system (including the EnergyPlus name, and humidity control).
    """
    # create openstudio ideal air system object
    os_ideal_air = OSZoneHVACIdealLoadsAirSystem(os_model)
    if room is None:
        os_ideal_air.setName(hvac.identifier)
    else:
        os_ideal_air.setName('{} Ideal Loads Air System'.format(room.identifier))
    if hvac._display_name is not None:
        os_ideal_air.setDisplayName(hvac.display_name)
    # assign the dehumidification based on the room
    os_ideal_air.setDehumidificationControlType('None')  # default when no humidistat
    if room is not None:
        setpoint = room.properties.energy.setpoint
        if setpoint.humidifying_schedule is not None:
            os_ideal_air.setDehumidificationControlType('Humidistat')
            os_ideal_air.setHumidificationControlType('Humidistat')
    # assign the economizer type
    os_ideal_air.setOutdoorAirEconomizerType(hvac.economizer_type)
    # set the sensible and latent heat recovery
    if hvac.sensible_heat_recovery != 0:
        os_ideal_air.setSensibleHeatRecoveryEffectiveness(hvac.sensible_heat_recovery)
        os_ideal_air.setHeatRecoveryType('Sensible')
    else:
        os_ideal_air.setSensibleHeatRecoveryEffectiveness(0)
    if hvac.latent_heat_recovery != 0:
        os_ideal_air.setLatentHeatRecoveryEffectiveness(hvac.latent_heat_recovery)
        os_ideal_air.setHeatRecoveryType('Enthalpy')
    else:
        os_ideal_air.setLatentHeatRecoveryEffectiveness(0)
    # assign the demand controlled ventilation
    if hvac.demand_controlled_ventilation:
        os_ideal_air.setDemandControlledVentilationType('OccupancySchedule')
    else:
        os_ideal_air.setDemandControlledVentilationType('None')
    # set the heating and cooling supply air temperature
    os_ideal_air.setMaximumHeatingSupplyAirTemperature(hvac.heating_air_temperature)
    os_ideal_air.setMinimumCoolingSupplyAirTemperature(hvac.cooling_air_temperature)
    # assign limits to the system's heating capacity
    if hvac.heating_limit == no_limit:
        os_ideal_air.setHeatingLimit('NoLimit')
    else:
        os_ideal_air.setHeatingLimit('LimitCapacity')
    if hvac.heating_limit == autosize:
        os_ideal_air.autosizeMaximumSensibleHeatingCapacity()
    else:
        os_ideal_air.setMaximumSensibleHeatingCapacity(hvac.heating_limit)
    # assign limits to the system's cooling capacity
    if hvac.cooling_limit == no_limit:
        os_ideal_air.setCoolingLimit('NoLimit')
    else:
        os_ideal_air.setCoolingLimit('LimitFlowRateAndCapacity')
    if hvac.cooling_limit == autosize:
        os_ideal_air.autosizeMaximumTotalCoolingCapacity()
        os_ideal_air.autosizeMaximumCoolingAirFlowRate()
    else:
        os_ideal_air.setMaximumTotalCoolingCapacity(hvac.cooling_limit)
        os_ideal_air.autosizeMaximumCoolingAirFlowRate()
    # assign heating availability schedule
    if hvac.heating_availability is not None:
        os_schedule = os_model.getScheduleByName(hvac.heating_availability.identifier)
        if os_schedule.is_initialized():
            os_schedule = os_schedule.get()
            os_ideal_air.setHeatingAvailabilitySchedule(os_schedule)
    # assign cooling availability schedule
    if hvac.cooling_availability is not None:
        os_schedule = os_model.getScheduleByName(hvac.cooling_availability.identifier)
        if os_schedule.is_initialized():
            os_schedule = os_schedule.get()
            os_ideal_air.setCoolingAvailabilitySchedule(os_schedule)
    return os_ideal_air


def ideal_air_system_from_openstudio(os_hvac, schedules=None):
    """Convert OpenStudio ZoneHVACIdealLoadsAirSystem to Honeybee IdealAirSystem."""
    hvac = IdealAirSystem(clean_ep_string(os_hvac.nameString()))
    hvac.economizer_type = os_hvac.outdoorAirEconomizerType()
    hvac.demand_controlled_ventilation = False \
        if os_hvac.demandControlledVentilationType().lower() in ('none', '') else True
    hvac.sensible_heat_recovery = os_hvac.sensibleHeatRecoveryEffectiveness()
    hvac.latent_heat_recovery = os_hvac.latentHeatRecoveryEffectiveness()
    hvac.heating_air_temperature = os_hvac.maximumHeatingSupplyAirTemperature()
    hvac.cooling_air_temperature = os_hvac.minimumCoolingSupplyAirTemperature()
    if not os_hvac.isHeatingLimitDefaulted():
        if os_hvac.heatingLimit().lower() == 'nolimit':
            hvac.heating_limit = no_limit
        elif os_hvac.heatingLimit().lower() == 'limitcapacity':
            if os_hvac.isMaximumSensibleHeatingCapacityAutosized():
                hvac.heating_limit = autosize
            elif os_hvac.maximumSensibleHeatingCapacity().is_initialized():
                hvac.heating_limit = os_hvac.maximumSensibleHeatingCapacity().get()
    if not os_hvac.isCoolingLimitDefaulted():
        if os_hvac.coolingLimit().lower() == 'nolimit':
            hvac.cooling_limit = no_limit
        elif os_hvac.coolingLimit().lower() == 'limitcapacity':
            if hvac.isMaximumTotalCoolingCapacityAutosized():
                hvac.cooling_limit = autosize
            elif os_hvac.maximumTotalCoolingCapacity().is_initialized():
                hvac.cooling_limit = os_hvac.maximumTotalCoolingCapacity()
    if schedules is not None and os_hvac.heatingAvailabilitySchedule().is_initialized():
        schedule = os_hvac.heatingAvailabilitySchedule().get()
        try:
            hvac.heating_availability = schedules[schedule.nameString()]
        except KeyError:
            pass
    if schedules is not None and os_hvac.coolingAvailabilitySchedule().is_initialized():
        schedule = os_hvac.coolingAvailabilitySchedule().get()
        try:
            hvac.cooling_availability = schedules[schedule.nameString()]
        except KeyError:
            pass
    if os_hvac.displayName().is_initialized():
        hvac.display_name = os_hvac.displayName().get()
    return hvac
