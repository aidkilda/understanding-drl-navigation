from envs.grid_2d_env.ColumnObservation import ColumnObservation
from envs.grid_2d_env.DepthObservation import DepthObservation
from envs.grid_2d_env.PositionAndRotationObservation import PositionAndRotationObservation
from envs.grid_2d_env.Action import Action
from envs.grid_2d_env.Cell import Cell
from envs.grid_2d_env.MazeGrid import MazeGrid
from envs.grid_2d_env.SingleLandmarkGrid import SingleLandmarkGrid
from utils.maze_generator import generate_random_maze_cells
from envs.grid_2d_env.HandmadeGrids import make_topological_setup, make_metric_triangle_setup

import tensorflow as tf
import random
import numpy as np

LOCAL_T_MAX = 5  # repeat step size
RMSP_ALPHA = 0.99  # decay parameter for RMSProp
RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp

INITIAL_ALPHA_LOW = 1e-4  # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2  # log_uniform high limit for learning rate
INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)

GAMMA = 0.99  # discount factor for rewards
ENTROPY_BETA = 0.01  # entropy regularization constant
MAX_TIME_STEP = 10.0 * 10**6  # 10 million frames
GRAD_NORM_CLIP = 40.0  # gradient norm clipping

ACTION_SIZE = 4  # action size
USE_LSTM = True
HISTORY_LENGTH = 1
HIDDEN_NEURONS = 60 # Number of neurons in hidden layers of the network.

PARALLEL_SIZE = 2 # parallel thread size
NUM_EVAL_EPISODES = 100  # number of episodes for evaluation

SEED = 11234567
USE_GPU = True  # To use GPU, set True

CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

shape = (5,5)

task_list = None
metric, task_list = make_metric_triangle_setup(eval=False)
#grid_cells = generate_random_maze_cells(shape[1], shape[0])

config = {
    'name': "metric",

    'model':"A3C",
    'task_list': task_list,

    # Reward for reaching target.
    'target_reach_reward':10,
    # Reward per agent's step (negative).
    'step_reward': -0.1,
    # the limit on the number of steps on the episode before it terminates.
    'episode_limit': 1000,
    # the list of actions that agent can take in the environment. See Action module for more detailed
    # explanation on what action does. Default actions are: 1)Go up 2) Go down 3) Turn left 4) Turn right.
    'actions': [
        Action((0, 1), 0),
        Action((0, -1), 0),
        Action((0, 0), 90),
        Action((0, 0), -90)
    ],
    # tells whether actions that do not change the agent state (e.g bumping into wall, or
    # trying to move out of bounds) are allowed.
    'allow_null_actions': True,
    # shape of the grid of the form (width, height).
    'shape': shape,

    'grid_class': MazeGrid,
    'initial_grid_cells': None,
    'random_reset_grid': False,

    # SingleLandmarkGrid
    'num_landmarks': 1,
    'target_distance_from_beacon': 1,
    'beacon_loc': None,

    'initial_agent': None,
    'random_reset_agent': False,

    'initial_target': None,
    'random_reset_target':False,
    'can_step_on_target': True,
    'is_target_transparent': False,
    # Target could be hidden, and, for example, identified only by nearby landmarks.
    'is_target_visible': True,
    'target_value': None,

    # Agent setup
    'observation_class': DepthObservation,
    'rotations': [0, 90, 180, 270],

    # Depth observation
    'angle': 120,
    'num_rays': 60,

    # Column observation
    'height': shape[0],
    'width': shape[1],

    'history_length': HISTORY_LENGTH
}
