# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/standards/PumpVariableSpeed.rb
"""
from __future__ import division


def pump_variable_speed_set_control_type(pump_variable_speed, control_type):
    """Set the pump curve coefficients based on the specified control type.

    Args:
        pump_variable_speed: [OpenStudio::Model::PumpVariableSpeed] variable speed pump.
        control_type: [String] valid choices are Riding Curve, VSD No Reset, VSD DP Reset.
    """
    # Determine the coefficients
    if control_type == 'Constant Flow':
        coeff_a = 0.0
        coeff_b = 1.0
        coeff_c = 0.0
        coeff_d = 0.0
    elif control_type == 'Riding Curve':
        coeff_a = 0.0
        coeff_b = 3.2485
        coeff_c = -4.7443
        coeff_d = 2.5294
    elif control_type == 'VSD No Reset':
        coeff_a = 0.0
        coeff_b = 0.5726
        coeff_c = -0.301
        coeff_d = 0.7347
    elif control_type == 'VSD DP Reset':
        coeff_a = 0.0
        coeff_b = 0.0205
        coeff_c = 0.4101
        coeff_d = 0.5753
    else:
        msg = 'Pump control type {} not recognized, pump coefficients will not ' \
            'be changed.'.format(control_type)
        print(msg)
        return None

    # Set the coefficients
    pump_variable_speed.setCoefficient1ofthePartLoadPerformanceCurve(coeff_a)
    pump_variable_speed.setCoefficient2ofthePartLoadPerformanceCurve(coeff_b)
    pump_variable_speed.setCoefficient3ofthePartLoadPerformanceCurve(coeff_c)
    pump_variable_speed.setCoefficient4ofthePartLoadPerformanceCurve(coeff_d)
    pump_variable_speed.setPumpControlType('Intermittent')
    return True
