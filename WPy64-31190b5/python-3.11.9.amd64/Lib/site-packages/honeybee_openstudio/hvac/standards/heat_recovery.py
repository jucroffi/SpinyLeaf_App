# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/hvac/components/create.rb
"""
from __future__ import division

from honeybee_openstudio.openstudio import openstudio_model


def create_hx_air_to_air_sensible_and_latent(
        model, name=None, type=None, economizer_lockout=None,
        supply_air_outlet_temperature_control=None, frost_control_type=None,
        nominal_electric_power=None,
        sensible_heating_100_eff=None, latent_heating_100_eff=None,
        sensible_cooling_100_eff=None, latent_cooling_100_eff=None):
    """Creates HeatExchangerAirToAirSensibleAndLatent object."""
    hx = openstudio_model.HeatExchangerAirToAirSensibleAndLatent(model)
    if name is not None:
        hx.setName(name)

    if type is not None:
        hx.setHeatExchangerType(type)

    if frost_control_type is not None:
        hx.setFrostControlType(frost_control_type)

    if economizer_lockout is not None:
        hx.setEconomizerLockout(economizer_lockout)
    if supply_air_outlet_temperature_control is not None:
        hx.setSupplyAirOutletTemperatureControl(supply_air_outlet_temperature_control)
    if nominal_electric_power is not None:
        hx.setNominalElectricPower(nominal_electric_power)

    if sensible_heating_100_eff is not None:
        hx.setSensibleEffectivenessat100HeatingAirFlow(sensible_heating_100_eff)
    if latent_heating_100_eff is not None:
        hx.setLatentEffectivenessat100HeatingAirFlow(latent_heating_100_eff)
    if sensible_cooling_100_eff is not None:
        hx.setSensibleEffectivenessat100CoolingAirFlow(sensible_cooling_100_eff)
    if latent_cooling_100_eff is not None:
        hx.setLatentEffectivenessat100CoolingAirFlow(latent_cooling_100_eff)

    return hx
