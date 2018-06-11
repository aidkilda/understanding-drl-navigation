import numpy as np

from .Observation import Observation


class PositionAndRotationObservation(Observation):
    """Represents the observation of the agent, that consists of it's position and rotation in the grid.

    This observation could be used as a sanity check, before trying more complicated observations.
    Beware that observation might not work without standardization/normalization.
    """

    def __init__(self):
        pass

    def get(self, agent_position_and_rotation, grid):
        return np.asarray([agent_position_and_rotation.x,
                           agent_position_and_rotation.y,
                           agent_position_and_rotation.rotation])

    def get_cells(self, agent_position_and_rotation, grid):
        return [agent_position_and_rotation.get_pos()]

    def size(self):
        return 3

    @classmethod
    def create(cls, config):
        """Use @classmethod polymorphism to be able to construct Observation Objects Generically.

        :param config: configuration dictionary for arguments used to set up the right observation.
        :return: constructed PositionObservation Object.
        """
        return cls()
