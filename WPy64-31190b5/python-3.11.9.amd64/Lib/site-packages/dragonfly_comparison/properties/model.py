# coding=utf-8
"""Model Comparison Properties."""
from dragonfly.extensionutil import model_extension_dicts


class ModelComparisonProperties(object):
    """Comparison Properties for Dragonfly Model.

    Args:
        host: A dragonfly_core Model object that hosts these properties.

    Properties:
        * host
    """

    def __init__(self, host):
        """Initialize Model Comparison properties."""
        self._host = host

    @property
    def host(self):
        """Get the Model object hosting these properties."""
        return self._host

    def set_from_model(self, comparison_model, reset_unmatched=True):
        """Set the attributes of Room2DComparisonProperties using another Model.

        Args:
            comparison_model: A dragonfly Model to which the host Model is
                being compared. Room2Ds in the host model with identifiers
                matching the comparison_model model will have their comparison
                properties set using them.
            reset_unmatched: A boolean to note whether rooms in the host model
                should have their comparison room properties reset if they are not
                matched with any room in the comparison_model. (Default: True).
        """
        for base_room in self.host.room_2ds:
            for comp_room in comparison_model.room_2ds:
                if base_room.identifier == comp_room.identifier:
                    base_room.properties.comparison.set_from_room_2d(comp_room)
                    break
            else:
                if reset_unmatched:
                    base_room.properties.comparison.reset()

    def reset(self):
        """Reset the comparison attributes using the host Model."""
        for base_room in self.host.room_2ds:
            base_room.properties.comparison.reset()

    def apply_properties_from_dict(self, data):
        """Apply the comparison properties of a dictionary to the host Model of this object.

        Args:
            data: A dictionary representation of an entire dragonfly-core Model.
                Note that this dictionary must have ModelComparisonProperties in order
                for this method to successfully apply the comparison properties.
        """
        assert 'comparison' in data['properties'], \
            'Dictionary possesses no ModelComparisonProperties.'
        # collect lists of comparison property dictionaries
        _, _, room2d_c_dicts, _ = \
            model_extension_dicts(data, 'comparison', [], [], [], [])
        # apply comparison properties to objects using the comparison property dictionaries
        for room, r_dict in zip(self.host.room_2ds, room2d_c_dicts):
            if r_dict is not None:
                room.properties.comparison.apply_properties_from_dict(r_dict)

    def to_dict(self):
        """Return Model comparison properties as a dictionary."""
        return {'comparison': {'type': 'ModelComparisonProperties'}}

    def duplicate(self, new_host=None):
        """Get a copy of this Model.

        Args:
            new_host: A new Model object that hosts these properties.
                If None, the properties will be duplicated with the same host.
        """
        _host = new_host or self._host
        return ModelComparisonProperties(_host)

    def ToString(self):
        return self.__repr__()

    def __repr__(self):
        return 'Model Comparison Properties: {}'.format(self.host.identifier)
