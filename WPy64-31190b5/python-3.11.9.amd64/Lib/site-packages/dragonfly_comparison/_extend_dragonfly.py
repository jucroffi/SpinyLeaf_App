# coding=utf-8
from dragonfly.properties import ModelProperties, Room2DProperties

from .properties.model import ModelComparisonProperties
from .properties.room2d import Room2DComparisonProperties


# set a hidden comparison attribute on each core geometry Property class to None
# define methods to produce comparison property instances on each Property instance
ModelProperties._comparison = None
Room2DProperties._comparison = None


def model_comparison_properties(self):
    if self._comparison is None:
        self._comparison = ModelComparisonProperties(self.host)
    return self._comparison


def room2d_comparison_properties(self):
    if self._comparison is None:
        self._comparison = Room2DComparisonProperties(self.host)
    return self._comparison


# add comparison property methods to the Properties classes
ModelProperties.comparison = property(model_comparison_properties)
Room2DProperties.comparison = property(room2d_comparison_properties)
