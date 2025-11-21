# coding=utf-8
"""OpenStudio material translators."""
from __future__ import division

from honeybee.typing import clean_ep_string
from honeybee_energy.material.opaque import EnergyMaterial, EnergyMaterialNoMass, \
    EnergyMaterialVegetation
from honeybee_energy.material.glazing import EnergyWindowMaterialGlazing, \
    EnergyWindowMaterialSimpleGlazSys
from honeybee_energy.material.gas import EnergyWindowMaterialGas, \
    EnergyWindowMaterialGasMixture, EnergyWindowMaterialGasCustom
from honeybee_energy.material.frame import EnergyWindowFrame
from honeybee_energy.material.shade import EnergyWindowMaterialShade, \
    EnergyWindowMaterialBlind

from honeybee_openstudio.openstudio import OSStandardOpaqueMaterial, \
    OSMasslessOpaqueMaterial, OSRoofVegetation, OSStandardGlazing, OSSimpleGlazing, \
    OSGas, OSGasMixture, OSShade, OSBlind, OSWindowPropertyFrameAndDivider


"""____________TRANSLATORS TO OPENSTUDIO____________"""


def opaque_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyMaterial to OpenStudio StandardOpaqueMaterial."""
    os_opaque_mat = OSStandardOpaqueMaterial(os_model)
    os_opaque_mat.setName(material.identifier)
    if material._display_name is not None:
        os_opaque_mat.setDisplayName(material.display_name)
    os_opaque_mat.setThickness(material.thickness)
    os_opaque_mat.setConductivity(material.conductivity)
    os_opaque_mat.setDensity(material.density)
    os_opaque_mat.setSpecificHeat(material.specific_heat)
    os_opaque_mat.setRoughness(material.roughness)
    os_opaque_mat.setThermalAbsorptance(material.thermal_absorptance)
    os_opaque_mat.setSolarAbsorptance(material.solar_absorptance)
    os_opaque_mat.setVisibleAbsorptance(material.visible_absorptance)
    return os_opaque_mat


def opaque_no_mass_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyMaterialNoMass to OpenStudio MasslessOpaqueMaterial."""
    os_nomass_mat = OSMasslessOpaqueMaterial(os_model)
    os_nomass_mat.setName(material.identifier)
    if material._display_name is not None:
        os_nomass_mat.setDisplayName(material.display_name)
    os_nomass_mat.setThermalResistance(material.r_value)
    os_nomass_mat.setRoughness(material.roughness)
    os_nomass_mat.setThermalAbsorptance(material.thermal_absorptance)
    os_nomass_mat.setSolarAbsorptance(material.solar_absorptance)
    os_nomass_mat.setVisibleAbsorptance(material.visible_absorptance)
    return os_nomass_mat


def vegetation_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyMaterialVegetation to OpenStudio RoofVegetation."""
    os_veg_mat = OSRoofVegetation(os_model)
    os_veg_mat.setName(material.identifier)
    if material._display_name is not None:
        os_veg_mat.setDisplayName(material.display_name)
    os_veg_mat.setThickness(material.thickness)
    os_veg_mat.setConductivityofDrySoil(material.conductivity)
    os_veg_mat.setDensityofDrySoil(material.density)
    os_veg_mat.setSpecificHeatofDrySoil(material.specific_heat)
    os_veg_mat.setRoughness(material.roughness)
    os_veg_mat.setThermalAbsorptance(material.soil_thermal_absorptance)
    os_veg_mat.setSolarAbsorptance(material.soil_solar_absorptance)
    os_veg_mat.setVisibleAbsorptance(material.soil_visible_absorptance)
    os_veg_mat.setHeightofPlants(material.plant_height)
    os_veg_mat.setLeafAreaIndex(material.leaf_area_index)
    os_veg_mat.setLeafReflectivity(material.leaf_reflectivity)
    os_veg_mat.setLeafEmissivity(material.leaf_emissivity)
    os_veg_mat.setMinimumStomatalResistance(material.min_stomatal_resist)
    os_veg_mat.setSaturationVolumetricMoistureContentoftheSoilLayer(material.sat_vol_moist_cont)
    os_veg_mat.setResidualVolumetricMoistureContentoftheSoilLayer(material.residual_vol_moist_cont)
    os_veg_mat.setInitialVolumetricMoistureContentoftheSoilLayer(material.init_vol_moist_cont)
    os_veg_mat.setMoistureDiffusionCalculationMethod(material.moist_diff_model)
    return os_veg_mat


def glazing_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialGlazing to OpenStudio StandardGlazing."""
    os_glazing = OSStandardGlazing(os_model)
    os_glazing.setName(material.identifier)
    if material._display_name is not None:
        os_glazing.setDisplayName(material.display_name)
    os_glazing.setThickness(material.thickness)
    os_glazing.setSolarTransmittanceatNormalIncidence(material.solar_transmittance)
    os_glazing.setFrontSideSolarReflectanceatNormalIncidence(material.solar_reflectance)
    os_glazing.setBackSideSolarReflectanceatNormalIncidence(material.solar_reflectance_back)
    os_glazing.setVisibleTransmittanceatNormalIncidence(material.visible_transmittance)
    os_glazing.setFrontSideVisibleReflectanceatNormalIncidence(material.visible_reflectance)
    os_glazing.setBackSideVisibleReflectanceatNormalIncidence(material.visible_reflectance_back)
    os_glazing.setInfraredTransmittanceatNormalIncidence(material.infrared_transmittance)
    os_glazing.setFrontSideInfraredHemisphericalEmissivity(material.emissivity)
    os_glazing.setBackSideInfraredHemisphericalEmissivity(material.emissivity_back)
    os_glazing.setThermalConductivity(material.conductivity)
    os_glazing.setDirtCorrectionFactorforSolarandVisibleTransmittance(material.dirt_correction)
    os_glazing.setSolarDiffusing(material.solar_diffusing)
    return os_glazing


def simple_glazing_sys_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialSimpleGlazSys to OpenStudio SimpleGlazing."""
    os_glz_sys = OSSimpleGlazing(os_model)
    os_glz_sys.setName(material.identifier)
    if material._display_name is not None:
        os_glz_sys.setDisplayName(material.display_name)
    os_glz_sys.setUFactor(material.u_factor)
    os_glz_sys.setSolarHeatGainCoefficient(material.shgc)
    os_glz_sys.setVisibleTransmittance(material.vt)
    return os_glz_sys


def gas_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialGas to OpenStudio Gas."""
    os_gas = OSGas(os_model)
    os_gas.setName(material.identifier)
    if material._display_name is not None:
        os_gas.setDisplayName(material.display_name)
    os_gas.setThickness(material.thickness)
    os_gas.setGasType(material.gas_type)
    return os_gas


def gas_mixture_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialGasMixture to OpenStudio GasMixture."""
    os_gas_mix = OSGasMixture(os_model)
    os_gas_mix.setName(material.identifier)
    if material._display_name is not None:
        os_gas_mix.setDisplayName(material.display_name)
    os_gas_mix.setThickness(material.thickness)
    for i in range(len(material.gas_types)):
        os_gas_mix.setGasType(i, material.gas_types[i])
        os_gas_mix.setGasFraction(i, material.gas_fractions[i])
    os_gas_mix.setNumberofGasesinMixture(len(material.gas_types))
    return os_gas_mix


def gas_custom_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialGasCustom to OpenStudio Gas."""
    os_gas_custom = OSGas(os_model)
    os_gas_custom.setName(material.identifier)
    if material._display_name is not None:
        os_gas_custom.setDisplayName(material.display_name)
    os_gas_custom.setThickness(material.thickness)
    os_gas_custom.setGasType('Custom')
    os_gas_custom.setConductivityCoefficientA(material.conductivity_coeff_a)
    os_gas_custom.setViscosityCoefficientA(material.viscosity_coeff_a)
    os_gas_custom.setSpecificHeatCoefficientA(material.specific_heat_coeff_a)
    os_gas_custom.setConductivityCoefficientB(material.conductivity_coeff_b)
    os_gas_custom.setViscosityCoefficientB(material.viscosity_coeff_b)
    os_gas_custom.setSpecificHeatCoefficientB(material.specific_heat_coeff_b)
    os_gas_custom.setConductivityCoefficientC(material.conductivity_coeff_c)
    os_gas_custom.setViscosityCoefficientC(material.viscosity_coeff_c)
    os_gas_custom.setSpecificHeatCoefficientC(material.specific_heat_coeff_c)
    os_gas_custom.setSpecificHeatRatio(material.specific_heat_ratio)
    os_gas_custom.setMolecularWeight(material.molecular_weight)
    return os_gas_custom


def shade_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialShade to OpenStudio Shade."""
    os_shade_mat = OSShade(os_model)
    os_shade_mat.setName(material.identifier)
    if material._display_name is not None:
        os_shade_mat.setDisplayName(material.display_name)
    os_shade_mat.setSolarTransmittance(material.solar_transmittance)
    os_shade_mat.setSolarReflectance(material.solar_reflectance)
    os_shade_mat.setVisibleTransmittance(material.visible_transmittance)
    os_shade_mat.setVisibleReflectance(material.visible_reflectance)
    os_shade_mat.setThermalHemisphericalEmissivity(material.emissivity)
    os_shade_mat.setThermalTransmittance(material.infrared_transmittance)
    os_shade_mat.setThickness(material.thickness)
    os_shade_mat.setConductivity(material.conductivity)
    os_shade_mat.setShadetoGlassDistance(material.distance_to_glass)
    os_shade_mat.setTopOpeningMultiplier(material.top_opening_multiplier)
    os_shade_mat.setBottomOpeningMultiplier(material.bottom_opening_multiplier)
    os_shade_mat.setLeftSideOpeningMultiplier(material.left_opening_multiplier)
    os_shade_mat.setRightSideOpeningMultiplier(material.right_opening_multiplier)
    os_shade_mat.setAirflowPermeability(material.airflow_permeability)
    return os_shade_mat


def blind_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowMaterialBlind to OpenStudio Blind."""
    os_blind = OSBlind(os_model)
    os_blind.setName(material.identifier)
    if material._display_name is not None:
        os_blind.setDisplayName(material.display_name)
    os_blind.setSlatOrientation(material.slat_orientation)
    os_blind.setSlatWidth(material.slat_width)
    os_blind.setSlatSeparation(material.slat_separation)
    os_blind.setSlatThickness(material.slat_thickness)
    os_blind.setSlatAngle(material.slat_angle)
    os_blind.setSlatConductivity(material.slat_conductivity)
    os_blind.setSlatBeamSolarTransmittance(material.beam_solar_transmittance)
    os_blind.setFrontSideSlatBeamSolarReflectance(material.beam_solar_reflectance)
    os_blind.setBackSideSlatBeamSolarReflectance(material.beam_solar_reflectance_back)
    os_blind.setSlatDiffuseSolarTransmittance(material.diffuse_solar_transmittance)
    os_blind.setFrontSideSlatDiffuseSolarReflectance(material.diffuse_solar_reflectance)
    os_blind.setBackSideSlatDiffuseSolarReflectance(material.diffuse_solar_reflectance_back)
    os_blind.setSlatDiffuseVisibleTransmittance(material.diffuse_visible_transmittance)
    os_blind.setFrontSideSlatDiffuseVisibleReflectance(material.diffuse_visible_reflectance)
    os_blind.setBackSideSlatDiffuseVisibleReflectance(material.diffuse_visible_reflectance_back)
    os_blind.setSlatBeamVisibleTransmittance(material.beam_visible_transmittance)
    os_blind.setFrontSideSlatBeamVisibleReflectance(material.beam_visible_reflectance)
    os_blind.setBackSideSlatBeamVisibleReflectance(material.beam_visible_reflectance_back)
    os_blind.setSlatInfraredHemisphericalTransmittance(material.infrared_transmittance)
    os_blind.setFrontSideSlatInfraredHemisphericalEmissivity(material.emissivity)
    os_blind.setBackSideSlatInfraredHemisphericalEmissivity(material.emissivity_back)
    os_blind.setBlindtoGlassDistance(material.distance_to_glass)
    os_blind.setBlindTopOpeningMultiplier(material.top_opening_multiplier)
    os_blind.setBlindBottomOpeningMultiplier(material.bottom_opening_multiplier)
    os_blind.setBlindLeftSideOpeningMultiplier(material.left_opening_multiplier)
    os_blind.setBlindRightSideOpeningMultiplier(material.right_opening_multiplier)
    return os_blind


def frame_material_to_openstudio(material, os_model):
    """Convert Honeybee EnergyWindowFrame to OpenStudio WindowPropertyFrameAndDivider."""
    os_frame_mat = OSWindowPropertyFrameAndDivider(os_model)
    os_frame_mat.setName(material.identifier)
    if material._display_name is not None:
        os_frame_mat.setDisplayName(material.display_name)
    os_frame_mat.setFrameWidth(material.width)
    os_frame_mat.setFrameConductance(material.conductance)
    os_frame_mat.setRatioOfFrameEdgeGlassConductanceToCenterOfGlassConductance(
        material.edge_to_center_ratio)
    os_frame_mat.setFrameOutsideProjection(material.outside_projection)
    os_frame_mat.setFrameInsideProjection(material.inside_projection)
    os_frame_mat.setFrameThermalHemisphericalEmissivity(material.thermal_absorptance)
    os_frame_mat.setFrameSolarAbsorptance(material.solar_absorptance)
    os_frame_mat.setFrameVisibleAbsorptance(material.visible_absorptance)
    return os_frame_mat


def material_to_openstudio(material, os_model):
    """Convert any Honeybee energy material into an OpenStudio object.

    Args:
        material: A honeybee-energy Python object of a material layer.
        os_model: The OpenStudio Model object to which the Room will be added.

    Returns:
        An OpenStudio object for the material.
    """
    if isinstance(material, EnergyMaterial):
        return opaque_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyMaterialNoMass):
        return opaque_no_mass_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyMaterialVegetation):
        return vegetation_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialGlazing):
        return glazing_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialSimpleGlazSys):
        return simple_glazing_sys_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialGas):
        return gas_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialGasMixture):
        return gas_mixture_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialGasCustom):
        return gas_custom_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowFrame):
        return frame_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialShade):
        return shade_material_to_openstudio(material, os_model)
    elif isinstance(material, EnergyWindowMaterialBlind):
        return blind_material_to_openstudio(material, os_model)
    else:
        raise ValueError(
            '{} is not a recognized Energy Material type'.format(type(material)))


"""____________TRANSLATORS FROM OPENSTUDIO____________"""


def opaque_material_from_openstudio(os_material):
    """Convert OpenStudio StandardOpaqueMaterial to Honeybee EnergyMaterial."""
    # create the material object
    thickness = os_material.thickness()
    conductivity = os_material.conductivity()
    density = os_material.density()
    specific_heat = os_material.specificHeat()
    material = EnergyMaterial(clean_ep_string(os_material.nameString()),
                              thickness, conductivity, density, specific_heat)
    # set the optional properties of the material
    _apply_roughness(os_material, material)
    material.thermal_absorptance = os_material.thermalAbsorptance()
    material.solar_absorptance = os_material.solarAbsorptance()
    material.visible_absorptance = os_material.visibleAbsorptance()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def opaque_no_mass_material_from_openstudio(os_material):
    """Convert OpenStudio MasslessOpaqueMaterial to Honeybee EnergyMaterialNoMass."""
    # create the material object
    r_value = os_material.thermalResistance()
    material = EnergyMaterialNoMass(clean_ep_string(os_material.nameString()), r_value)
    # set the optional properties of the material
    try:
        _apply_roughness(os_material, material)
    except AttributeError:
        return material  # OpenStudio AirGap material with no roughness
    if os_material.thermalAbsorptance().is_initialized():
        material.thermal_absorptance = os_material.thermalAbsorptance().get()
    if os_material.solarAbsorptance().is_initialized():
        material.solar_absorptance = os_material.solarAbsorptance().get()
    if os_material.visibleAbsorptance().is_initialized():
        material.visible_absorptance = os_material.visibleAbsorptance().get()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def vegetation_material_from_openstudio(os_material):
    """Convert OpenStudio RoofVegetation to Honeybee EnergyMaterialVegetation."""
    # create the material object
    thickness = os_material.thickness()
    conductivity = os_material.conductivityofDrySoil()
    density = os_material.densityofDrySoil()
    specific_heat = os_material.specificHeatofDrySoil()
    plant_height = os_material.heightofPlants()
    leaf_area_index = os_material.leafAreaIndex()
    leaf_reflectivity = os_material.leafReflectivity()
    leaf_emissivity = os_material.leafEmissivity()
    min_stomatal_resist = os_material.minimumStomatalResistance()
    material = EnergyMaterialVegetation(
        clean_ep_string(os_material.nameString()),
        thickness, conductivity, density, specific_heat,
        plant_height=plant_height, leaf_area_index=leaf_area_index,
        leaf_reflectivity=leaf_reflectivity, leaf_emissivity=leaf_emissivity,
        min_stomatal_resist=min_stomatal_resist)
    # set the other required properties
    material.sat_vol_moist_cont = os_material.saturationVolumetricMoistureContent()
    material.residual_vol_moist_cont = os_material.residualVolumetricMoistureContent()
    material.init_vol_moist_cont = os_material.initialVolumetricMoistureContent()
    material.moist_diff_model = os_material.moistureDiffusionCalculationMethod().title()
    # set the optional properties of the material
    _apply_roughness(os_material, material)
    if os_material.thermalAbsorptance().is_initialized():
        material.soil_thermal_absorptance = os_material.thermalAbsorptance().get()
    if os_material.solarAbsorptance().is_initialized():
        material.soil_solar_absorptance = os_material.solarAbsorptance().get()
    if os_material.visibleAbsorptance().is_initialized():
        material.soil_visible_absorptance = os_material.visibleAbsorptance().get()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def _apply_roughness(os_material, material):
    """Apply roughness from an OpenStudio Material to a Honeybee Material."""
    rough = os_material.roughness().lower()
    if rough == 'veryrough':
        material.roughness == 'VeryRough'
    elif rough == 'mediumrough':
        material.roughness == 'MediumRough'
    elif rough == 'mediumsmooth':
        material.roughness == 'MediumSmooth'
    elif rough == 'verysmooth':
        material.roughness == 'VerySmooth'
    else:  # Rough or Smooth
        material.roughness = rough.title()


def glazing_material_from_openstudio(os_material):
    """Convert OpenStudio StandardGlazing to Honeybee EnergyWindowMaterialGlazing."""
    if os_material.opticalDataType().lower() == 'spectral':
        return None
    # get the solar transmittance
    try:
        solar_transmittance = os_material.solarTransmittance()
    except Exception:  # spectral material to ignore
        return None
    # create the material objects
    thickness = os_material.thickness()
    conductivity = os_material.conductivity()
    infrared_transmittance = os_material.infraredTransmittance()
    emissivity = os_material.frontSideInfraredHemisphericalEmissivity()
    emissivity_back = os_material.backSideInfraredHemisphericalEmissivity()
    conductivity = os_material.thermalConductivity()
    material = EnergyWindowMaterialGlazing(
        clean_ep_string(os_material.nameString()), thickness, solar_transmittance,
        conductivity=conductivity, infrared_transmittance=infrared_transmittance,
        emissivity=emissivity, emissivity_back=emissivity_back)
    # set all of the optional properties
    if os_material.frontSideSolarReflectanceatNormalIncidence().is_initialized():
        material.solar_reflectance = \
            os_material.frontSideSolarReflectanceatNormalIncidence().get()
    if os_material.backSideSolarReflectanceatNormalIncidence().is_initialized():
        material.solar_reflectance_back = \
            os_material.backSideSolarReflectanceatNormalIncidence().get()
    if os_material.visibleTransmittanceatNormalIncidence().is_initialized():
        material.visible_transmittance = \
            os_material.visibleTransmittanceatNormalIncidence().get()
    if os_material.frontSideVisibleReflectanceatNormalIncidence().is_initialized():
        try:
            material.visible_reflectance = \
                os_material.frontSideVisibleReflectanceatNormalIncidence().get()
        except AssertionError:
            pass  # illegal combination of transmittance and reflectance
    if os_material.backSideVisibleReflectanceatNormalIncidence().is_initialized():
        try:
            material.visible_reflectance_back = \
                os_material.backSideVisibleReflectanceatNormalIncidence().get()
        except AssertionError:
            pass  # illegal combination of transmittance and reflectance
    material.dirt_correction = \
        os_material.dirtCorrectionFactorforSolarandVisibleTransmittance()
    material.solar_diffusing = os_material.solarDiffusing()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def simple_glazing_sys_material_from_openstudio(os_material):
    """Convert OpenStudio SimpleGlazing to Honeybee EnergyWindowMaterialSimpleGlazSys."""
    # create the material object
    u_factor = os_material.uFactor()
    shgc = os_material.solarHeatGainCoefficient()
    material = EnergyWindowMaterialSimpleGlazSys(
        clean_ep_string(os_material.nameString()), u_factor, shgc)
    # set the optional properties of the material
    if os_material.visibleTransmittance().is_initialized():
        material.vt = os_material.visibleTransmittance().get()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def gas_material_from_openstudio(os_material):
    """Convert OpenStudio Gas to Honeybee EnergyWindowMaterialGas."""
    # create the material object
    thickness = os_material.thickness()
    gas_type = os_material.gasType()
    material = EnergyWindowMaterialGas(
        clean_ep_string(os_material.nameString()), thickness, gas_type)
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def gas_mixture_material_from_openstudio(os_material):
    """Convert OpenStudio GasMixture to Honeybee EnergyWindowMaterialGasMixture."""
    thickness = os_material.thickness()
    gas_count = os_material.numberofGasesinMixture()
    gas_types = [os_material.gas1Type(), os_material.gas2Type()]
    gas_fractions = [os_material.gas1Fraction(), os_material.gas2Fraction()]
    if gas_count > 2:
        if os_material.gas3Fraction().is_initialized():
            gas_types.append(os_material.gas3Type())
            gas_fractions.append(os_material.gas3Fraction().get())
        if gas_count > 3:
            if os_material.gas4Fraction().is_initialized():
                gas_types.append(os_material.gas4Type())
                gas_fractions.append(os_material.gas4Fraction().get())
    material = EnergyWindowMaterialGasMixture(
        clean_ep_string(os_material.nameString()), thickness, gas_types, gas_fractions)
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def gas_custom_material_from_openstudio(os_material):
    """Convert OpenStudio Gas to Honeybee EnergyWindowMaterialGasCustom."""
    # create the material object
    thickness = os_material.thickness()
    conductivity_coeff_a, viscosity_coeff_a, specific_heat_coeff_a = 0, 0, 0
    if os_material.customConductivityCoefficientA().is_initialized():
        conductivity_coeff_a = os_material.customConductivityCoefficientA().get()
    if os_material.viscosityCoefficientA().is_initialized():
        viscosity_coeff_a = os_material.viscosityCoefficientA().get()
    if os_material.specificHeatCoefficientA().is_initialized():
        specific_heat_coeff_a = os_material.specificHeatCoefficientA().get()
    material = EnergyWindowMaterialGasCustom(
        clean_ep_string(os_material.nameString()), thickness,
        conductivity_coeff_a, viscosity_coeff_a, specific_heat_coeff_a)
    # set the optional properties of the material
    if os_material.customConductivityCoefficientB().is_initialized():
        material.conductivity_coeff_b = os_material.customConductivityCoefficientB().get()
    if os_material.viscosityCoefficientB().is_initialized():
        material.viscosity_coeff_b = os_material.viscosityCoefficientB().get()
    if os_material.specificHeatCoefficientB().is_initialized():
        material.specific_heat_coeff_b = os_material.specificHeatCoefficientB().get()
    if os_material.customConductivityCoefficientC().is_initialized():
        material.conductivity_coeff_c = os_material.customConductivityCoefficientC().get()
    if os_material.viscosityCoefficientC().is_initialized():
        material.viscosity_coeff_c = os_material.viscosityCoefficientC().get()
    if os_material.specificHeatCoefficientC().is_initialized():
        material.specific_heat_coeff_c = os_material.specificHeatCoefficientC().get()
    if os_material.specificHeatRatio().is_initialized():
        material.specific_heat_ratio = os_material.specificHeatRatio().get()
    if os_material.molecularWeight().is_initialized():
        material.molecular_weight = os_material.molecularWeight().get()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def shade_material_from_openstudio(os_material):
    """Convert OpenStudio Shade to Honeybee EnergyWindowMaterialShade."""
    # create the material object
    material = EnergyWindowMaterialShade(clean_ep_string(os_material.nameString()))
    # set the optional properties of the material
    material.solar_transmittance = os_material.solarTransmittance()
    material.solar_reflectance = os_material.solarReflectance()
    material.visible_transmittance = os_material.visibleTransmittance()
    material.visible_reflectance = os_material.visibleReflectance()
    material.emissivity = os_material.thermalHemisphericalEmissivity()
    material.infrared_transmittance = os_material.thermalTransmittance()
    material.thickness = os_material.thickness()
    material.conductivity = os_material.conductivity()
    material.distance_to_glass = os_material.shadetoGlassDistance()
    material.top_opening_multiplier = os_material.topOpeningMultiplier()
    material.bottom_opening_multiplier = os_material.bottomOpeningMultiplier()
    material.left_opening_multiplier = os_material.leftSideOpeningMultiplier()
    material.right_opening_multiplier = os_material.rightSideOpeningMultiplier()
    material.airflow_permeability = os_material.airflowPermeability()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def blind_material_from_openstudio(os_material):
    """Convert OpenStudio Blind to Honeybee EnergyWindowMaterialBlind."""
    # create the material object
    material = EnergyWindowMaterialBlind(clean_ep_string(os_material.nameString()))
    # set the optional properties of the material
    material.slat_orientation = os_material.slatOrientation()
    material.slat_width = os_material.slatWidth()
    material.slat_separation = os_material.slatSeparation()
    material.slat_thickness = os_material.slatThickness()
    material.slat_width = os_material.slatWidth()
    material.slat_angle = os_material.slatAngle()
    material.slat_conductivity = os_material.slatConductivity()
    material.beam_solar_transmittance = os_material.slatBeamSolarTransmittance()
    material.beam_solar_reflectance = os_material.frontSideSlatBeamSolarReflectance()
    material.beam_solar_reflectance_back = os_material.backSideSlatBeamSolarReflectance()
    material.diffuse_solar_reflectance = \
        os_material.frontSideSlatDiffuseSolarReflectance()
    material.diffuse_solar_reflectance_back = \
        os_material.backSideSlatDiffuseSolarReflectance()
    material.diffuse_visible_transmittance = \
        os_material.slatDiffuseVisibleTransmittance()
    material.infrared_transmittance = \
        os_material.slatInfraredHemisphericalTransmittance()
    material.emissivity = os_material.frontSideSlatInfraredHemisphericalEmissivity()
    material.emissivity_back = os_material.backSideSlatInfraredHemisphericalEmissivity()
    material.distance_to_glass = os_material.blindtoGlassDistance()
    material.top_opening_multiplier = os_material.blindTopOpeningMultiplier()
    material.bottom_opening_multiplier = os_material.blindBottomOpeningMultiplier()
    material.left_opening_multiplier = os_material.blindLeftSideOpeningMultiplier()
    material.right_opening_multiplier = os_material.blindRightSideOpeningMultiplier()
    if os_material.frontSideSlatDiffuseVisibleReflectance().is_initialized():
        material.diffuse_visible_reflectance = \
            os_material.frontSideSlatDiffuseVisibleReflectance().get()
    if os_material.backSideSlatDiffuseVisibleReflectance().is_initialized():
        material.diffuse_visible_reflectance_back = \
            os_material.backSideSlatDiffuseVisibleReflectance().get()
    if os_material.displayName().is_initialized():
        material.display_name = os_material.displayName().get()
    return material


def extract_all_materials(os_model):
    """Extract all material objects from an OpenStudio Model.

    Args:
        os_model: The OpenStudio Model object from which materials will be extracted.

    Returns:
        A dictionary of material objects with material identifiers as keys and
        material objects as values.
    """
    materials = {}
    for os_mat in os_model.getStandardOpaqueMaterials():
        mat = opaque_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getMasslessOpaqueMaterials():
        mat = opaque_no_mass_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getAirGaps():
        mat = opaque_no_mass_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getRoofVegetations():
        mat = vegetation_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getSimpleGlazings():
        mat = simple_glazing_sys_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getStandardGlazings():
        mat = glazing_material_from_openstudio(os_mat)
        if mat is not None:
            materials[mat.identifier] = mat
    for os_mat in os_model.getGass():
        if os_mat.gasType() == 'Custom':
            mat = gas_custom_material_from_openstudio(os_mat)
        else:
            mat = gas_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getGasMixtures():
        mat = gas_mixture_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getShades():
        mat = shade_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    for os_mat in os_model.getBlinds():
        mat = blind_material_from_openstudio(os_mat)
        materials[mat.identifier] = mat
    return materials
