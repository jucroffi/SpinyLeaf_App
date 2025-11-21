"""Handlers to convert various boolean options to boolean flag strings."""
from .helper import bool_option_to_str


def filter_des_days_to_str(value):
    """Translate a boolean to the filter_des_days flag.

        Args:
            value: Either a boolean or one of two text strings.

            * filter-des-days
            * all-des-days

        Returns:
            str -- filter_des_days flag text.
    """
    return bool_option_to_str(
        value, ('filter-des-days', 'all-des-days'), 'filter_des_days'
    )


def skip_overture_to_str(value):
    """Translate a boolean to the skip_overture flag.

        Args:
            value: Either a boolean or one of two text strings.

            * skip-overture
            * overture

        Returns:
            str -- skip_overture flag text.
    """
    return bool_option_to_str(value, ('skip-overture', 'overture'), 'skip_overture')


def glare_control_devices_to_str(value):
    """Translate a boolean to the glare_control_devices flag.

        Args:
            value: Either a boolean or one of two text strings.

            * glare-control
            * no-glare-control

        Returns:
            str -- glare_control_devices flag text.
    """
    return bool_option_to_str(
        value, ('glare-control', 'no-glare-control'), 'glare_control_devices'
    )


def use_multiplier_to_str(value):
    """Translate a boolean to the use_multiplier flag.

        Args:
            value: Either a boolean or one of two text strings.

            * multiplier
            * full-geometry

        Returns:
            str -- use_multiplier flag text.
    """
    return bool_option_to_str(value, ('multiplier', 'full-geometry'), 'use_multiplier')


def is_residential_to_str(value):
    """Translate a boolean to the is_residential flag.

        Args:
            value: Either a boolean or one of two text strings.

            * residential
            * nonresidential

        Returns:
            str -- is_residential flag text.
    """
    return bool_option_to_str(value, ('residential', 'nonresidential'), 'is_residential')


def write_set_map_to_str(value):
    """Translate a boolean to the write_set_map flag.

        Args:
            value: Either a boolean or one of two text strings.

            * write-op-map
            * write-set-map

        Returns:
            str -- write_set_map flag text.
    """
    return bool_option_to_str(value, ('write-set-map', 'write-op-map'), 'write_set_map')


def sky_view_metric_to_str(value):
    """Translate a boolean to the sky view metric flag.

        Args:
            value: Either a boolean or one of two text strings.

            * sky-exposure
            * sky-view

        Returns:
            str -- sky view metric flag text.
    """
    return bool_option_to_str(value, ('sky-exposure', 'sky-view'), 'metric')


def cloudy_bool_to_str(value):
    """Translate a boolean to the uniform/cloudy flag.

        Args:
            value: Either a boolean or one of two text strings.

            * cloudy
            * uniform

        Returns:
            str -- sky view uniform/cloudy flag text.
    """
    return bool_option_to_str(value, ('cloudy', 'uniform'), 'metric')


def visible_vs_solar_to_str(value):
    """Translate a boolean to the visible vs solar flag.

        Args:
            value: Either a boolean or one of two text strings.

            * visible
            * solar

        Returns:
            str -- visible vs solar flag text.
    """
    return bool_option_to_str(value, ('visible', 'solar'), 'visible')


def bldg_lighting_to_str(value):
    """Translate a boolean to the lighting-method flag of Appendix G simulations.

        Args:
            value: Either a boolean or one of two text strings.

            * building
            * space

        Returns:
            str -- building vs space flag text.
    """
    return bool_option_to_str(value, ('building', 'space'), 'bldg_lighting')
