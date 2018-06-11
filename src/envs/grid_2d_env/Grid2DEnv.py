"""Grid2DEnv is a 2-dimensional grid environment for reinforcement learning navigational tasks."""

from .SingleAgentEnv import SingleAgentEnv
from .Action import Action
from .Grid2DModel import Grid2DModel
from .Grid2DModelRenderer import Grid2DModelRenderer


class Grid2DEnv(SingleAgentEnv):
    """ Represents a 2-dimensional grid environment."""

    def __init__(self, config):
        """Initialise environment.

        :param config: configuration dictionary for arguments used to set up the right environment.
            (grid, agent, target, observation).
        """
        self.config = config

        self.shape = config['shape']
        self.episode_limit = config['episode_limit']
        self.actions = config['actions']
        self.allow_null_actions = config['allow_null_actions']

        self.grid_model = Grid2DModel(self.config)

        # Rest of initialization is done in reset method.
        self.reset()

    def reset(self):
        """Resets/Initially sets up parts the environment."""
        self.steps = 0

        self.grid_model.reset()
        self.renderer = Grid2DModelRenderer(self.grid_model)

        return self.get_obs()

    def step(self, action_index):
        """Perform one step in the environment corresponding to a given action.

        :param action_index: The index in the list of self.actions, of the action that should be performed by the agent
            during the step.
        :return: reward, has_terminated, info
        """
        action = self.actions[action_index]
        self.agent = self.grid_model.update_agent(action)
        reward = self.__get_reward()

        self.steps += 1
        has_terminated = self.__game_terminated()

        return reward, has_terminated

    def get_obs(self):
        """
        :return: the observation.
        """
        return self.grid_model.get_obs()

    def get_obs_size(self):
        """
        :return: the shape of the observation.
        """
        return self.grid_model.get_obs_size()

    def get_grid(self):
        return self.grid_model.grid

    def get_agent_state(self):
        """
        :return: agent's state (position and rotation).
        """
        return self.grid_model.get_agent_state()

    def get_target(self):
        return self.grid_model.target

    def get_avail_actions(self):
        """

        :return: the actions that are available for the agent. If null_actions are allowed, then all actions are
            available. Otherwise null_actions are not included amongst available actions.
        """
        if not self.allow_null_actions:
            return [1, 1, 1, 1]

        avail_actions = [0] * self.get_total_actions()
        for i, a in enumerate(self.actions):
            if not self.grid_model.is_null_action(a):
                avail_actions[i] = 1
        return avail_actions

    def render(self):
        self.renderer.render()

    def close(self):
        self.renderer.close()

    def __get_reward(self):
        """
        :return: the reward of the current state of the agent.
        """
        if self.grid_model.reached_target():
            return self.config['target_reach_reward']
        return self.config['step_reward']

    def __game_terminated(self):
        """
        :return: whether the game has terminated.

        The game is considered to be terminated if the agent reaches the target,
        or if the number of steps have exceeded the limit.
        """
        if self.steps >= self.episode_limit \
           or self.grid_model.reached_target():
            return True
        return False
