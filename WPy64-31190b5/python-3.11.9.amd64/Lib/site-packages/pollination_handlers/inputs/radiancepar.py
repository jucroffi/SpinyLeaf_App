"""Handlers for validating Radiance parameters."""
from honeybee_radiance_command.options.rpict import RpictOptions
from honeybee_radiance_command.options.rtrace import RtraceOptions
from honeybee_radiance_command.options.rcontrib import RcontribOptions
from honeybee_radiance_command.options.rfluxmtx import RfluxmtxOptions


def validate_rpict_params(input_params):
    rpict_options = RpictOptions()
    rpict_options.update_from_string(input_params, raise_error=True)
    return input_params

def validate_rtrace_params(input_params):
    rtrace_options = RtraceOptions()
    rtrace_options.update_from_string(input_params, raise_error=True)
    return input_params


def validate_rcontrib_params(input_params):
    rcontrib_options = RcontribOptions()
    rcontrib_options.update_from_string(input_params, raise_error=True)
    return input_params


def validate_rfluxmtx_params(input_params):
    rfluxmtx_options = RfluxmtxOptions()
    rfluxmtx_options.update_from_string(input_params, raise_error=True)
    return input_params
