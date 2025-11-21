# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.utilities.rb
"""
from __future__ import division

import re

from ladybug.datatype.power import Power

from honeybee_openstudio.openstudio import openstudio_model

POWER = Power()


def kw_per_ton_to_cop(kw_per_ton):
    """A helper method to convert from kW/ton to COP."""
    return 3.517 / kw_per_ton


def eer_to_cop_no_fan(eer, capacity_w=None):
    """Convert from EER to COP.

    If capacity is not supplied, use DOE Prototype Building method.
    If capacity is supplied, use the 90.1-2013 method.

    Args:
        eer: [Double] Energy Efficiency Ratio (EER).
        capacity_w: [Double] the heating capacity at AHRI rating conditions, in W.
    """
    if capacity_w is None:
        # From Thornton et al. 2011
        # r is the ratio of supply fan power to total equipment power at the rating condition,
        # assumed to be 0.12 for the reference buildings per Thornton et al. 2011.
        r = 0.12
        cop = ((eer / POWER.to_unit([1.0], 'Btu/h', 'W')[0]) + r) / (1 - r)
    else:
        # The 90.1-2013 method
        # Convert the capacity to Btu/hr
        capacity_btu_per_hr = POWER.to_unit([capacity_w], 'Btu/h', 'W')[0]
        cop = (7.84E-8 * eer * capacity_btu_per_hr) + (0.338 * eer)
    return cop


def hspf_to_cop_no_fan(hspf):
    """Convert from HSPF to COP (no fan) for heat pump heating coils.

    References - ASHRAE 90.1-2013. Appendix G.

    Args:
        hspf: [Double] heating seasonal performance factor (HSPF).
    """
    cop = (-0.0296 * hspf * hspf) + (0.7134 * hspf)
    return cop


def ems_friendly_name(name):
    """Converts existing string to ems friendly string."""
    # replace white space and special characters with underscore
    # \W is equivalent to [^a-zA-Z0-9_]
    new_name = re.sub('[^A-Za-z0-9]', '_', str(name))
    # prepend ems_ in case the name starts with a number
    new_name = 'ems_{}'.format(new_name)
    return new_name


def convert_curve_biquadratic(coeffs, ip_to_si=True):
    """Convert biquadratic curves that are a function of temperature.

    From IP (F) to SI (C) or vice-versa.  The curve is of the form
    z = C1 + C2*x + C3*x^2 + C4*y + C5*y^2 + C6*x*y
    where C1, C2, ... are the coefficients,
    x is the first independent variable (in F or C)
    y is the second independent variable (in F or C)
    and z is the resulting value
    """
    if ip_to_si:
        # Convert IP curves to SI curves
        si_coeffs = []
        si_coeffs.append((coeffs[0] + (32.0 * (coeffs[1] + coeffs[3])) +
                          (1024.0 * (coeffs[2] + coeffs[4] + coeffs[5]))))
        si_coeffs.append(((9.0 / 5.0 * coeffs[1]) +
                          (576.0 / 5.0 * coeffs[2]) + (288.0 / 5.0 * coeffs[5])))
        si_coeffs.append((81.0 / 25.0 * coeffs[2]))
        si_coeffs.append(((9.0 / 5.0 * coeffs[3]) +
                          (576.0 / 5.0 * coeffs[4]) + (288.0 / 5.0 * coeffs[5])))
        si_coeffs.append((81.0 / 25.0 * coeffs[4]))
        si_coeffs.append((81.0 / 25.0 * coeffs[5]))
        return si_coeffs
    else:
        # Convert SI curves to IP curves
        ip_coeffs = []
        ip_coeffs.append((coeffs[0] - (160.0 / 9.0 * (coeffs[1] + coeffs[3])) +
                          (25600.0 / 81.0 * (coeffs[2] + coeffs[4] + coeffs[5]))))
        ip_coeffs.append((5.0 / 9.0 * (coeffs[1] - (320.0 / 9.0 * coeffs[2]) -
                                       (160.0 / 9.0 * coeffs[5]))))
        ip_coeffs.append((25.0 / 81.0 * coeffs[2]))
        ip_coeffs.append((5.0 / 9.0 * (coeffs[3] - (320.0 / 9.0 * coeffs[4]) -
                                       (160.0 / 9.0 * coeffs[5]))))
        ip_coeffs.append((25.0 / 81.0 * coeffs[4]))
        ip_coeffs.append((25.0 / 81.0 * coeffs[5]))
        return ip_coeffs


def create_curve_biquadratic(
        model, coeffs, crv_name, min_x, max_x, min_y, max_y, min_out, max_out):
    """Create a biquadratic curve."""
    curve = openstudio_model.CurveBiquadratic(model)
    curve.setName(crv_name)
    curve.setCoefficient1Constant(coeffs[0])
    curve.setCoefficient2x(coeffs[1])
    curve.setCoefficient3xPOW2(coeffs[2])
    curve.setCoefficient4y(coeffs[3])
    curve.setCoefficient5yPOW2(coeffs[4])
    curve.setCoefficient6xTIMESY(coeffs[5])
    if min_x is None:
        curve.setMinimumValueofx(min_x)
    if max_x is not None:
        curve.setMaximumValueofx(max_x)
    if min_y is not None:
        curve.setMinimumValueofy(min_y)
    if max_y is not None:
        curve.setMaximumValueofy(max_y)
    if min_out is not None:
        curve.setMinimumCurveOutput(min_out)
    if max_out is not None:
        curve.setMaximumCurveOutput(max_out)
    return curve


def create_curve_quadratic(
        model, coeffs, crv_name, min_x, max_x, min_out, max_out, is_dimensionless=False):
    """Create a quadratic curve."""
    curve = openstudio_model.CurveQuadratic(model)
    curve.setName(crv_name)
    curve.setCoefficient1Constant(coeffs[0])
    curve.setCoefficient2x(coeffs[1])
    curve.setCoefficient3xPOW2(coeffs[2])
    if min_x is None:
        curve.setMinimumValueofx(min_x)
    if max_x is not None:
        curve.setMaximumValueofx(max_x)
    if min_out is not None:
        curve.setMinimumCurveOutput(min_out)
    if max_out is not None:
        curve.setMaximumCurveOutput(max_out)
    if is_dimensionless:
        curve.setInputUnitTypeforX('Dimensionless')
        curve.setOutputUnitType('Dimensionless')
    return curve


def rename_air_loop_nodes(model):
    """Renames air loop nodes to readable values."""
    # rename all hvac components on air loops
    for component in model.getHVACComponents():
        if component.to_Node().is_initialized():  # don't re-rename the node
            continue

        if component.airLoopHVAC().is_initialized():
            # rename water to air component outlet nodes
            if component.to_WaterToAirComponent().is_initialized():
                component = component.to_WaterToAirComponent().get()
                if component.airOutletModelObject().is_initialized():
                    component_outlet_object = component.airOutletModelObject().get()
                    if not component_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_outlet_object.setName('{} Outlet Air Node'.format(cp_name))

            # rename air to air component nodes
            if component.to_AirToAirComponent().is_initialized():
                component = component.to_AirToAirComponent().get()
                if component.primaryAirOutletModelObject().is_initialized():
                    component_outlet_object = component.primaryAirOutletModelObject().get()
                    if not component_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_outlet_object.setName(
                        '{} Primary Outlet Air Node'.format(cp_name))
                if component.secondaryAirInletModelObject().is_initialized():
                    component_inlet_object = component.secondaryAirInletModelObject().get()
                    if not component_inlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_inlet_object.setName(
                        '{} Secondary Inlet Air Node'.format(cp_name))

            # rename straight component outlet nodes
            if component.to_StraightComponent().is_initialized():
                st_comp = component.to_StraightComponent().get()
                if st_comp.outletModelObject().is_initialized():
                    component_outlet_object = st_comp.outletModelObject().get()
                    if not component_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_outlet_object.setName('{} Outlet Air Node'.format(cp_name))

        # rename zone hvac component nodes
        if component.to_ZoneHVACComponent().is_initialized():
            component = component.to_ZoneHVACComponent().get()
            if component.airInletModelObject().is_initialized():
                component_inlet_object = component.airInletModelObject().get()
                if not component_inlet_object.to_Node().is_initialized():
                    continue
                cp_name = component.nameString()
                component_inlet_object.setName('{} Inlet Air Node'.format(cp_name))

            if component.airOutletModelObject().is_initialized():
                component_outlet_object = component.airOutletModelObject().get()
                if not component_outlet_object.to_Node().is_initialized():
                    continue
                cp_name = component.nameString()
                component_outlet_object.setName('{} Outlet Air Node'.format(cp_name))

    # rename supply side nodes
    for air_loop in model.getAirLoopHVACs():
        air_loop_name = air_loop.nameString()
        air_loop.demandInletNode().setName('{} Demand Inlet Node'.format(air_loop_name))
        air_loop.demandOutletNode().setName('{} Demand Outlet Node'.format(air_loop_name))
        air_loop.supplyInletNode().setName('{} Supply Inlet Node'.format(air_loop_name))
        air_loop.supplyOutletNode().setName('{} Supply Outlet Node'.format(air_loop_name))

        if air_loop.reliefAirNode().is_initialized():
            relief_node = air_loop.reliefAirNode().get()
            relief_node.setName('{} Relief Air Node'.format(air_loop_name))

        if air_loop.mixedAirNode().is_initialized():
            mixed_node = air_loop.mixedAirNode().get()
            mixed_node.setName('{} Mixed Air Node'.format(air_loop_name))

        # rename outdoor air system and nodes
        if air_loop.airLoopHVACOutdoorAirSystem().is_initialized():
            oa_system = air_loop.airLoopHVACOutdoorAirSystem().get()
            if oa_system.outboardOANode().is_initialized():
                oa_node = oa_system.outboardOANode().get()
                oa_node.setName('{} Outdoor Air Node'.format(air_loop_name))

    # rename zone air and terminal nodes
    for zone in model.getThermalZones():
        zone_name = zone.nameString()
        zone.zoneAirNode().setName('{} Zone Air Node'.format(zone_name))
        if zone.returnAirModelObject().is_initialized():
            zone_node = zone.returnAirModelObject().get()
            zone_node.setName('{} Return Air Node'.format(zone_name))

        if zone.airLoopHVACTerminal().is_initialized():
            terminal_unit = zone.airLoopHVACTerminal().get()
            unit_name = terminal_unit.nameString()
            if terminal_unit.to_StraightComponent().is_initialized():
                component = terminal_unit.to_StraightComponent().get()
                if component.inletModelObject().is_initialized():
                    in_node = component.inletModelObject().get()
                    in_node.setName('{} Inlet Air Node'.format(unit_name))

    # rename zone equipment list objects
    for obj in model.getZoneHVACEquipmentLists():
        try:
            zone = obj.thermalZone()
            obj.setName('{} Zone HVAC Equipment List'.format(zone.nameString()))
        except Exception:
            obj.remove()  # missing thermal zone

    return model


def rename_plant_loop_nodes(model):
    """Renames plant loop nodes to readable values."""
    # rename all hvac components on plant loops
    for component in model.getHVACComponents():
        if component.to_Node().is_initialized():  # don't re-rename the node
            continue

        if component.plantLoop().is_initialized():
            # rename straight component nodes
            # some inlet or outlet nodes may get renamed again
            if component.to_StraightComponent().is_initialized():
                st_comp = component.to_StraightComponent().get()
                if st_comp.inletModelObject().is_initialized():
                    component_inlet_object = st_comp.inletModelObject().get()
                    if not component_inlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_inlet_object.setName(
                        '{} Inlet Water Node'.format(cp_name))

                if st_comp.outletModelObject().is_initialized():
                    component_outlet_object = st_comp.outletModelObject().get()
                    if not component_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_outlet_object.setName(
                        '{} Outlet Water Node'.format(cp_name))

            # rename water to air component nodes
            if component.to_WaterToAirComponent().is_initialized():
                component = component.to_WaterToAirComponent().get()
                if component.waterInletModelObject().is_initialized():
                    component_inlet_object = component.waterInletModelObject().get()
                    if not component_inlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_inlet_object.setName(
                        '{} Inlet Water Node'.format(cp_name))

                if component.waterOutletModelObject().is_initialized():
                    component_outlet_object = component.waterOutletModelObject().get()
                    if not component_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    component_outlet_object.setName(
                        '{} Outlet Water Node'.format(cp_name))

            # rename water to water component nodes
            if component.to_WaterToWaterComponent().is_initialized():
                component = component.to_WaterToWaterComponent().get()
                if component.demandInletModelObject().is_initialized():
                    demand_inlet_object = component.demandInletModelObject().get()
                    if not demand_inlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    demand_inlet_object.setName(
                        '{} Demand Inlet Water Node'.format(cp_name))

                if component.demandOutletModelObject().is_initialized():
                    demand_outlet_object = component.demandOutletModelObject().get()
                    if not demand_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    demand_outlet_object.setName(
                        '{} Demand Outlet Water Node'.format(cp_name))

                if component.supplyInletModelObject().is_initialized():
                    supply_inlet_object = component.supplyInletModelObject().get()
                    if not supply_inlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    supply_inlet_object.setName(
                        '{} Supply Inlet Water Node'.format(cp_name))

                if component.supplyOutletModelObject().is_initialized():
                    supply_outlet_object = component.supplyOutletModelObject().get()
                    if not supply_outlet_object.to_Node().is_initialized():
                        continue
                    cp_name = component.nameString()
                    supply_outlet_object.setName(
                        '{} Supply Outlet Water Node'.format(cp_name))

    # rename plant nodes
    for plant_loop in model.getPlantLoops():
        pl_name = plant_loop.nameString()
        plant_loop.demandInletNode().setName('{} Demand Inlet Node'.format(pl_name))
        plant_loop.demandOutletNode().setName('{} Demand Outlet Node'.format(pl_name))
        plant_loop.supplyInletNode().setName('{} Supply Inlet Node'.format(pl_name))
        plant_loop.supplyOutletNode().setName('{} Supply Outlet Node'.format(pl_name))

    return model
