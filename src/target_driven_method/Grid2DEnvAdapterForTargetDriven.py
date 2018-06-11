"""
Grid2DEnv has a different API to AI2-THOR environment, which is used by the original target-driven navigation method
code. Thus, to be able to reuse as much of that code as possible, the adapter from grid2denv API
to AI2-THOR API is implemented.
"""

from envs.grid_2d_env.Grid2DEnv import Grid2DEnv
from envs.grid_2d_env.PositionAndRotation import PositionAndRotation
from utils.scale_features import normalize, standardize
import numpy as np


class Grid2DEnvAdapter(object):

    def __init__(self, config):

        rotations = config['rotations']

        self.grid_env = Grid2DEnv(config)

        self.empty_cells = self.__get_empty_cells()
        self.locations = np.repeat(self.empty_cells, len(rotations), axis=0)
        self.n_locations = self.locations.shape[0]
        self.rotations = np.tile(rotations, (self.n_locations // len(rotations), 1)).flatten()

        self.all_observations = np.asarray([self.get_observation(id) for id in range(len(self.locations))])
        self.observations_mean = self.all_observations.mean(axis=0)
        self.observations_std = self.all_observations.std(axis=0)
        self.observations_max_values = self.all_observations.max(axis=0)
        self.observations_min_values = self.all_observations.min(axis=0)

        self.terminal_state_id = self.__get_state_id(self.grid_env.get_target().get_pos(), 0)

        self.terminals = np.zeros(self.n_locations)
        self.terminals[self.terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)

        self.history_length = config['history_length']

        self.s_t = np.zeros([self.grid_env.get_obs_size(), self.history_length])
        self.s_t1 = np.zeros_like(self.s_t)

        self.s_target = self.tiled_state(self.terminal_state_id)

        self.reset()

    def reset(self):
        self.grid_env.reset()
        self.__update_current_state_id()

        self.terminal_state_id = self.__get_state_id(self.grid_env.get_target().get_pos(), 0)

        self.terminals = np.zeros(self.n_locations)
        self.terminals[self.terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)

        self.start_state_id = self.current_state_id
        self.s_t = self.tiled_state(self.current_state_id)

        self.colided = False
        self.terminal = False

    def step(self, action_index):
        self.reward, self.terminal = self.grid_env.step(action_index)
        self.__update_current_state_id()
        self.s_t = np.append(self.s_t[:, 1:], self.state, axis=1)

    def get_observation(self, state_id):
        x = self.get_x(state_id)
        y = self.get_y(state_id)
        rotation = self.get_rotation(state_id)
        pos_and_rot = PositionAndRotation(x, y, rotation)
        grid = self.grid_env.get_grid()
        return self.grid_env.grid_model.agent.observation.get(pos_and_rot, grid)

    def get_state(self, state_id):
        return self.tiled_state(state_id)

    def get_x(self, state_id):
        return self.locations[state_id][0]

    def get_y(self, state_id):
        return self.locations[state_id][1]

    def get_rotation(self, state_id):
        return self.rotations[state_id]

    def render(self):
        self.grid_env.render()

    def __get_empty_cells(self):
        empty_cells = []
        grid = self.grid_env.get_grid()
        for r in range(grid.rows):
            for c in range(grid.cols):
                # Add if a cell is not a wall cell.
                if not grid.cells[r][c].value == 1:
                    empty_cells.append([r,c])
        return empty_cells

    def __update_current_state_id(self):
        agent_state = self.grid_env.get_agent_state()
        agent_pos = agent_state.get_pos()
        agent_rotation = agent_state.rotation
        self.current_state_id = self.__get_state_id(agent_pos, agent_rotation)

    def __get_state_id(self, position, rotation):
        locations = self.locations.tolist()
        i = 0
        for index, loc in enumerate(locations):
            if loc[0] == position[0] and loc[1] == position[1]:
                i = index
                break

        if rotation == 90:
            i += 1
        elif rotation == 180:
            i += 2
        elif rotation == 270:
            i += 3
        return i

    def tiled_state(self, state_id):
        # Need to be able to get observation given the position&rotation (or state_id).
        agent_pos_rot = PositionAndRotation(self.locations[state_id][0],
                                            self.locations[state_id][1],
                                            self.rotations[state_id])
        grid = self.grid_env.grid_model.grid
        obs = self.grid_env.grid_model.agent.observation.get(agent_pos_rot, grid)
        std_obs = self.__standardize_observation(obs)
        non_tiled_state = std_obs[:, np.newaxis]
        return np.tile(non_tiled_state, (1, self.history_length))

    def __normalize_observation(self, obs):
        return normalize(obs, self.observations_max_values, self.observations_min_values)

    def __standardize_observation(self, obs):
        return standardize(obs, self.observations_mean, self.observations_std)

    # properties

    @property
    def action_size(self):
        # move forward/backward, turn left/right for navigation
        return self.grid_env.get_avail_actions()

    @property
    def observation(self):
        return self.grid_env.get_obs()

    @property
    def obs_size(self):
        return self.grid_env.get_obs_size()

    @property
    def state(self):
        return self.__standardize_observation(self.grid_env.get_obs())[:, np.newaxis]

    @property
    def target(self):
        return self.s_target

    @property
    def x(self):
        return self.locations[self.current_state_id][0]

    @property
    def y(self):
        return self.locations[self.current_state_id][1]

    @property
    def r(self):
        return self.rotations[self.current_state_id]


