from .Observation import Observation
from .VisibleCellsFinder import VisibleCellsFinder
from utils.list_utils import flatten

import numpy as np


class DepthObservation(Observation):
    def __init__(self, angle, num_rays):
        """Initialize DepthObservation.

        :param angle: agent's viewing angle.
        :param num_rays: number of rays, along which the number of empty cells is being checked.
        """
        self.angle = angle
        self.num_rays = num_rays

        self.visible_cells_finder = VisibleCellsFinder()

    def get(self, agent_position_and_rotation, grid):
        ray_angles = self.__get_all_ray_angles()
        depth_obs = self.__get_depth_vector(agent_position_and_rotation, grid, ray_angles)
        landmark_obs = self.__get_landmark_vector(agent_position_and_rotation, grid, ray_angles)
        return np.asarray(depth_obs + landmark_obs, dtype=np.float32)

    def get_cells(self, agent_position_and_rotation, grid):
        """ Returns grid cells, that are observed by the agent. Note that list might contain out of bounds cells."""
        ray_angles = self.__get_all_ray_angles()
        return flatten(self.visible_cells_finder.get_visible_cells_along_rays(
            grid, ray_angles, agent_position_and_rotation))

    def size(self):
        return 2 * self.num_rays

    @classmethod
    def create(cls, config):
        """Use @classmethod polymorphism to be able to construct Observation Objects Generically.

        :param config: configuration dictionary for arguments used to set up the right observation.
        :return: constructed DepthObservation Object.
        """
        angle = config['angle']
        num_rays = config['num_rays']
        return cls(angle, num_rays)

    def __get_all_ray_angles(self):
        first_angle = -self.angle / 2
        diff_between_ray_angles = self.angle
        if self.num_rays > 1:
            diff_between_ray_angles = self.angle / (self.num_rays - 1)
        return [
            first_angle + i * diff_between_ray_angles
            for i in range(self.num_rays)
        ]

    def __get_depth_vector(self, agent_position_and_rotation, grid, ray_angles):
        return self.visible_cells_finder.get_num_visible_cells_along_rays(
            grid, ray_angles, agent_position_and_rotation)


    def __get_landmark_vector(self, agent_position_and_rotation, grid, ray_angles):
        visible_cells_along_rays = self.visible_cells_finder.get_visible_cells_along_rays(
            grid, ray_angles, agent_position_and_rotation)
        last_visible_cells_along_rays = [cells[-1] for cells in visible_cells_along_rays]
        return [grid.get_location_value(cell[0], cell[1]) for cell in last_visible_cells_along_rays]