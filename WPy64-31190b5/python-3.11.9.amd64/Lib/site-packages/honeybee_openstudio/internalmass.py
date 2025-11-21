# coding=utf-8
"""OpenStudio internal mass translator."""
from __future__ import division
from honeybee_openstudio.openstudio import OSInternalMassDefinition, OSInternalMass


def internal_mass_to_openstudio(internal_mass, os_model):
    """Convert Honeybee InternalMass to OpenStudio InternalMass."""
    os_mass_def = OSInternalMassDefinition(os_model)
    os_mass_def.setName(internal_mass.identifier)
    os_mass_def.setSurfaceArea(internal_mass.area)
    construction_id = internal_mass.construction.identifier
    os_construction = os_model.getConstructionByName(construction_id)
    if os_construction.is_initialized():
        os_construction = os_construction.get()
        os_mass_def.setConstruction(os_construction)
    os_mass = OSInternalMass(os_mass_def)
    os_mass.setName(internal_mass.identifier)
    return os_mass
