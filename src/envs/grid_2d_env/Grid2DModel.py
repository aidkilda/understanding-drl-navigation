"""Grid2DModel is a model for 2-dimensional grid environment for reinforcement learning navigational tasks.

Grid2DModel encapsulates the grid itself, the target and the agent.
"""

from .Agent import Agent
from .Target import Target

import copy

class Grid2DModel:
    """Grid2DModel represents the grid, target and agent."""

    def __init__(self, config):
        """Initializes Grid2DModel.

        :param config: configuration dictionary for arguments used to set up the right agent and right observation.
        """
        self.config = config
        self.shape = config['shape']

        self.grid = config['grid_class'].create(config)

        self.target = Target(config, self.grid)
        self.is_target_visible = config['is_target_visible']
        if self.is_target_visible:
            self.grid.set_target(self.target.get_pos(),
                                 self.config['can_step_on_target'],
                                 self.config['is_target_transparent'],
                                 self.config['target_value'])

        self.agent = Agent(config, self.grid, self.target)

    def reset(self):
        """Resets the Grid2DModel to initial state."""
        self.grid.reset()

        self.target.reset(self.grid)
        if self.is_target_visible:
            self.grid.set_target(self.target.get_pos(),
                                 self.config['can_step_on_target'],
                                 self.config['is_target_transparent'],
                                 self.config['target_value'])

        self.agent.reset(self.grid, self.target)

    def update_agent(self, action):
        """Updates the state of the agent after performing the given action.

        :param action: action to be performed by the agent.
        """
        new_pos_and_rot = self.agent.position_and_rotation_after_action(action)
        new_x, new_y = new_pos_and_rot.get_pos()
        if self.grid.can_step_location(new_x, new_y):
            self.agent.update_agent(new_pos_and_rot)

    def get_obs(self):
        """
        :return: the observation of the agent.
        """
        return self.agent.get_obs(self.grid)

    def get_obs_cells(self):
        """Get the cells, that are observed by the agent. This method is useful when visualizing environment and agent's movement in it.

        :return: cells, observed by the agent.
        """
        return self.agent.get_obs_cells(self.grid)

    def get_obs_size(self):
        """
        :return: the shape of the observation.
        """
        return self.agent.get_obs_size()

    def get_agent_state(self):
        """
        :return: agent's state (position and rotation).
        """
        return self.agent.position_and_rotation

    def reached_target(self):
        """
        :return: whether agent has reached the target.
        """
        return self.agent.get_pos() == self.target.get_pos()

    def is_null_action(self, action):
        """
        :param action: Action that can be performed by the agent.
        :return: whether the given action does not change the agent's state.
        """
        return self.agent.position_and_rotation_after_action(
            action) == self.agent.get_state()