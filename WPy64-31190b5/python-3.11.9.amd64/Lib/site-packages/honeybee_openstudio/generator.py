# coding=utf-8
"""OpenStudio photovoltaic and generator translators."""
from __future__ import division

from honeybee_openstudio.openstudio import OSGeneratorPVWatts, \
    OSElectricLoadCenterDistribution, OSElectricLoadCenterInverterPVWatts


def pv_properties_to_openstudio(pv_properties, os_shade, os_model):
    """Convert Honeybee PVProperties to OpenStudio GeneratorPVWatts.

    Args:
        pv_properties: The Honeybee PVProperties object to be translated
            to OpenStudio.
        os_shade: The OpenStudio ShadingSurface object to which the GeneratorPVWatts
            is being assigned.
        os_model: The OpenStudio model to which the GeneratorPVWatts
            will be added.
    """
    # compte the system capacity
    rated_watts = pv_properties.rated_efficiency * 1000  # 1000W/m2 of solar irradiance
    sys_cap = int(os_shade.netArea() * pv_properties.active_area_fraction * rated_watts)
    # create the PVWatts generator and set all of the properties
    os_gen = OSGeneratorPVWatts(os_model, os_shade, sys_cap)
    os_gen.setName('{}..{}'.format(pv_properties.identifier, os_shade.nameString()))
    os_gen.setModuleType(pv_properties.module_type)
    os_gen.setArrayType(pv_properties.mounting_type)
    os_gen.setSystemLosses(pv_properties.system_loss_fraction)
    os_gen.setGroundCoverageRatio(pv_properties.tracking_ground_coverage_ratio)
    return os_gen


def electric_load_center_to_openstudio(load_center, os_gen_objects, os_model):
    """Convert Honeybee ElectricLoadCenter to OpenStudio ElectricLoadCenterDistribution.

    Args:
        load_center: The Honeybee ElectricLoadCenter object to be translated
            to OpenStudio.
        os_gen_objects: A list of the OpenStudio Generators objects that are
            controlled by the ElectricLoadCenter.
        os_model: The OpenStudio model to which the ElectricLoadCenterDistribution
            will be added.
    """
    # create the ElectricLoadCenter:Distribution and add the generators
    os_load_center = OSElectricLoadCenterDistribution(os_model)
    os_load_center.setName('Model Load Center Distribution')
    for os_gen in os_gen_objects:
        os_load_center.addGenerator(os_gen)
    os_load_center.setGeneratorOperationSchemeType('Baseload')
    os_load_center.setElectricalBussType('DirectCurrentWithInverter')
    # create the inverter and assign it
    inverter = OSElectricLoadCenterInverterPVWatts(os_model)
    inverter.setDCToACSizeRatio(load_center.inverter_dc_to_ac_size_ratio)
    inverter.setInverterEfficiency(load_center.inverter_efficiency)
    os_load_center.setInverter(inverter)
    return os_load_center
