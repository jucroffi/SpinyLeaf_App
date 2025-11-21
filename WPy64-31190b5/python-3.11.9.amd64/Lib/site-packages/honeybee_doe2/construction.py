"""honeybee-inp construction translators."""
from __future__ import division

from ladybug.datatype.uvalue import UValue
from ladybug.datatype.rvalue import RValue
from ladybug.datatype.distance import Distance
from honeybee.typing import clean_doe2_string
from honeybee_energy.material.opaque import EnergyMaterialNoMass

from .config import RES_CHARS, MIN_LAYER_THICKNESS
from .util import generate_inp_string, generate_inp_string_list_format

# dictionary to map between E+ and DOE-2 roughness types
ROUGHNESS_MAP = {
    'VeryRough': 1, 'Rough': 2, 'MediumRough': 3,
    'MediumSmooth': 4, 'Smooth': 5, 'VerySmooth': 6
}


def opaque_material_to_inp(material):
    """Convert an EnergyMaterial or EnergyMaterialNoMass into a MATERIAL INP string.

    Note that EnergyMaterials that are below a certain thickness will be automatically
    converted to No Mass materials for compatibility with DOE-2. Also note that
    this does not work for any materials that can be a part of a window constructions.
    """
    doe2_id = clean_doe2_string(material.identifier, RES_CHARS)
    # check if the material should be translated as a no mass material
    if isinstance(material, EnergyMaterialNoMass) or \
            material.thickness < MIN_LAYER_THICKNESS:
        r_val = RValue().to_unit([material.r_value], 'h-ft2-F/Btu', 'm2-K/W')[0]
        keywords = ('TYPE', 'RESISTANCE')
        values = ('RESISTANCE', round(r_val, 6))
        return generate_inp_string(doe2_id, 'MATERIAL', keywords, values)
    # write out detailed properties for the material
    thickness = round(Distance().to_unit([material.thickness], 'ft', 'm')[0], 3)
    conduct = round(material.conductivity * 0.578176, 3)  # convert to BTU/h-ft-F
    density = round(material.density / 16.018, 3)  # convert to lb/ft3
    spec_en = round(material.specific_heat * 0.0002388459, 3)  # convert to BTU/lb-F
    keywords = ('TYPE', 'THICKNESS', 'CONDUCTIVITY', 'DENSITY', 'SPECIFIC-HEAT')
    values = ('PROPERTIES', thickness, conduct, density, spec_en)
    return generate_inp_string(doe2_id, 'MATERIAL', keywords, values)


def opaque_construction_to_inp(construction):
    """Convert an OpaqueConstruction into a CONSTRUCTION INP string.

    This will include both the LAYERS definition as well as the CONSTRUCTION but
    it does NOT include the constituent MATERIAL definitions and their properties.
    """
    doe2_id = clean_doe2_string(construction.identifier, RES_CHARS)
    # if the construction has no heat capacity, simply make a U-VALUE construction
    if construction.area_heat_capacity == 0 or \
            construction.thickness <= MIN_LAYER_THICKNESS * len(construction.materials):
        con_cond = UValue().to_unit([construction.u_factor], 'Btu/h-ft2-F', 'W/m2-K')[0]
        keywords = ('TYPE', 'U-VALUE')
        values = ('U-VALUE', round(con_cond, 6))
        return generate_inp_string(doe2_id, 'CONSTRUCTION', keywords, values)
    # create the specification of material layers
    layer_id = '{}_l'.format(doe2_id)
    layers = ['"{}"'.format(clean_doe2_string(mat, RES_CHARS))
              for mat in construction.layers]
    layer_str = generate_inp_string_list_format(
        layer_id, 'LAYERS', ['MATERIAL'], [layers])
    # create the construction specification
    roughness = ROUGHNESS_MAP[construction.materials[0].roughness]
    sol_absorb = round(1 - construction.outside_solar_reflectance, 3)
    keywords = ('TYPE', 'ABSORPTANCE', 'ROUGHNESS', 'LAYERS')
    values = ('LAYERS', sol_absorb, roughness, '"{}"'.format(layer_id))
    constr_str = generate_inp_string(doe2_id, 'CONSTRUCTION', keywords, values)
    return ''.join((layer_str, constr_str))


def window_construction_to_inp(construction):
    """Convert a WindowConstruction (or its variants) into a GLASS-TYPE INP string."""
    doe2_id = clean_doe2_string(construction.identifier, RES_CHARS)
    shading_coef = construction.shgc / 0.87
    glass_cond = UValue().to_unit([construction.u_factor], 'Btu/h-ft2-F', 'W/m2-K')[0]
    keywords = ('TYPE', 'SHADING-COEF', 'GLASS-CONDUCT')
    values = ('SHADING-COEF', round(shading_coef, 3), round(glass_cond, 6))
    return generate_inp_string(doe2_id, 'GLASS-TYPE', keywords, values)


def door_construction_to_inp(construction):
    """Convert an OpaqueConstruction or WindowConstruction to a CONSTRUCTION INP string.

    This translation pathway always uses a NoMass U-VALUE Construction.
    """
    doe2_id = clean_doe2_string(construction.identifier, RES_CHARS)
    constr_cond = UValue().to_unit([construction.u_factor], 'Btu/h-ft2-F', 'W/m2-K')[0]
    keywords = ('TYPE', 'U-VALUE')
    values = ('U-VALUE', round(constr_cond, 6))
    return generate_inp_string(doe2_id, 'CONSTRUCTION', keywords, values)


def air_construction_to_inp(construction):
    """Convert an AirBoundaryConstruction to a CONSTRUCTION INP string.

    This translation pathway always uses a NoMass U-VALUE Construction.
    """
    doe2_id = clean_doe2_string(construction.identifier, RES_CHARS)
    constr_cond = 1.0  # default U-Value in Btu/h-ft2-F
    keywords = ('TYPE', 'U-VALUE')
    values = ('U-VALUE', constr_cond)
    return generate_inp_string(doe2_id, 'CONSTRUCTION', keywords, values)
