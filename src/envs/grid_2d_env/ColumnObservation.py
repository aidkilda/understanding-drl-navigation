from .Observation import Observation
from .VisibleCellsFinder import VisibleCellsFinder

from utils.list_utils import flatten

import numpy as np


class ColumnObservation(Observation):
    """Represents the observation of the agent, that consists of pixels in front of the agent.

    The size of observation vector is max(h, w), where h and w are height and width of the grid, where agent navigates,
    respectively.
    If the view of the agent is obscured by walls, then all pixel values behind the first wall are treated as wall cells.
    """

    def __init__(self, height, width):
        """ Initialize ColumnObservation.

        :param height: height of the grid, where agent navigates.
        :param width: width of the grid, where agent navigates.
        """
        self.obs_len = max(height, width)

        self.visible_cells_finder = VisibleCellsFinder()

    def get(self, agent_position_and_rotation, grid):
        # Only the angle of 0 degrees is used.
        angles = [0]
        visible_cells = flatten(self.visible_cells_finder.get_visible_cells_along_rays(
            grid, angles, agent_position_and_rotation))
        # Initialize observation as if all cells are unknown, i.e., are wall cells.
        agent_obs = [1] * self.obs_len
        for i, cell in enumerate(visible_cells):
            if not grid.is_out_of_bounds_wall_location(cell[0], cell[1]):
                agent_obs[i] = grid.get_location_value(cell[0], cell[1])
        return np.asarray(agent_obs, dtype=np.float32)

    def get_cells(self, agent_position_and_rotation, grid):
        """ Returns grid cells, that are observed by the agent. Note that list might contain out of bounds cells."""
        angles = [0]
        return flatten(self.visible_cells_finder.get_visible_cells_along_rays(
            grid, angles, agent_position_and_rotation))

    def size(self):
        return self.obs_len

    @classmethod
    def create(cls, config):
        """Use @classmethod polymorphism to be able to construct Observation Objects Generically.

        :param config: configuration dictionary for arguments used to set up the right observation.
        :return: constructed ColumnObservation Object.
        """
        height = config['height']
        width = config['width']
        return cls(height, width)
