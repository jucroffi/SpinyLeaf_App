"""Functions for post-processing annual irradiance outputs."""

from ladybug.datatype.energyflux import EnergyFlux
from ladybug.datatype.energyintensity import EnergyIntensity
from ladybug.legend import LegendParameters


def _annual_irradiance_vis_metadata():
    """Return visualization metadata for annual irradiance."""
    cumulative_radiation_lpar = LegendParameters(min=0)
    peak_irradiance_lpar = LegendParameters(min=0)
    average_irradiance_lpar = LegendParameters(min=0)

    metric_info_dict = {
        'cumulative_radiation': {
            'type': 'VisualizationMetaData',
            'data_type': EnergyIntensity('Cumulative Radiance').to_dict(),
            'unit': 'kWh/m2',
            'legend_parameters': cumulative_radiation_lpar.to_dict()
        },
        'peak_irradiance': {
            'type': 'VisualizationMetaData',
            'data_type': EnergyFlux('Peak Irradiance').to_dict(),
            'unit': 'W/m2',
            'legend_parameters': peak_irradiance_lpar.to_dict()
        },
        'average_irradiance': {
            'type': 'VisualizationMetaData',
            'data_type': EnergyFlux('Average Irradiance').to_dict(),
            'unit': 'W/m2',
            'legend_parameters': average_irradiance_lpar.to_dict()
        }
    }

    return metric_info_dict
