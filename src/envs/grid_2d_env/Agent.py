"""Agent encapsulates the concept of an agent(state and observation) in the environment."""

from .PositionAndRotation import PositionAndRotation
import numpy as np
import copy


#TODO(aidkilda) abstract out agent's state to separate class.
class Agent:
    """Represents agent in the environment."""

    def __init__(self, config, grid, target):
        """

        :param config: configuration dictionary for arguments used to set up the right agent and right observation.
        :param grid: grid where agent navigates.
        :param target: target that agent attempts to reach.
            It is only used to avoid initializing agent to the position where the target is.
        """
        self.config = config

        self.observation = config['observation_class'].create(config)
        self.position_and_rotation = config['initial_agent']

        # If no position and rotation is passed, then initialize it randomly.
        if not config['initial_agent']:
            self.position_and_rotation = self.__random_position_and_rotation(grid, target)

        self.random_reset = config['random_reset_agent']
        self.initial_position_and_rotation = copy.deepcopy(self.position_and_rotation)

    def reset(self, grid, target):
        """Resets the agent's position and rotation. Note that it depends on the current grid and target."""
        if self.random_reset:
            self.position_and_rotation = self.__random_position_and_rotation(grid, target)
        else:
            self.position_and_rotation = copy.deepcopy(self.initial_position_and_rotation)

    def position_and_rotation_after_action(self, action):
        """
        :param action: Action to be performed by the agent.
        :return: what agent's position and rotation would be after the action, represented by PositionAndRotation object.
        """
        return action.perform(self.position_and_rotation)

    def update_agent(self, new_position_and_rotation):
        """Update agent's state(position and rotation)."""
        self.position_and_rotation = new_position_and_rotation

    def get_pos(self):
        """
        :return: postion of the agent (x, y) coordinates.
        """
        return self.position_and_rotation.get_pos()

    def get_obs(self, grid):
        """
        :param grid: the grid, where agent acts.
        :return: agent's observation.
        """
        return self.observation.get(self.position_and_rotation, grid)

    def get_obs_cells(self, grid):
        """Get the cells, that are observed by the agent. This method is useful when visualizing environment and agent's movement in it.

        :return: cells, observed by the agent.
        """
        return self.observation.get_cells(self.position_and_rotation, grid)

    def get_obs_size(self):
        """
        :return: the shape of the observation.
        """
        return self.observation.size()

    def get_state(self):
        """
        :return: numpy array representing agent's state (position and rotation).
        """
        return np.asarray(self.position_and_rotation.to_list())

    def __random_position_and_rotation(self, grid, target):
        rotations = self.config['rotations']
        x, y = grid.get_random_empty_location()
        # Avoid agent's position from coinciding with target's position.
        while (x, y) == target.get_pos():
            x, y = grid.get_random_empty_location()

        rotation = np.random.choice(rotations)
        return PositionAndRotation(x, y, rotation)
