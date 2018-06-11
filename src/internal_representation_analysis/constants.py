# -*- coding: utf-8 -*-

LOCAL_T_MAX = 5  # repeat step size
RMSP_ALPHA = 0.99  # decay parameter for RMSProp
RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'logs'
INITIAL_ALPHA_LOW = 1e-4  # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2  # log_uniform high limit for learning rate

PARALLEL_SIZE = 4  # parallel thread size
ACTION_SIZE = 4  # action size

INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99  # discount factor for rewards
ENTROPY_BETA = 0.01  # entropy regurarlization constant
MAX_TIME_STEP = 10.0 * 10**6  # 10 million frames
GRAD_NORM_CLIP = 40.0  # gradient norm clipping
USE_GPU = True  # To use GPU, set True
VERBOSE = False

SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4

NUM_EVAL_EPISODES = 100  # number of episodes for evaluation

TASK_TYPE = 'navigation'  # no need to change

# keys are scene names, and values are a list of location ids (navigation targets)
TASK_LIST = {
    'bathroom_02': ['26', '37', '43', '53', '69'],
    'bedroom_04': ['134', '264', '320', '384', '387'],
    'kitchen_02': ['90', '136', '157', '207', '329'],
    'living_room_08': ['92', '135', '193', '228', '254']
}

# Constants for position decoder:

EPOCHS = 5000
MINI_BATCH_SIZE = 100

DROPOUT_KEEP_PROB = 0.5
DROPOUT_KEEP_PROB_POLAR_ANGLE = 0.05
DROPOUT_KEEP_PROB_POS_AND_ANGLE = 1
DROPOUT_KEEP_PROB_ANGLE_TO_TARGET = 0.1
DROPOUT_KEEP_PROB_LOOKING_ANGLE = 0.1

ADAM_LEARNING_RATE = 0.00001
ADAM_LEARNING_RATE_POLAR_ANGLE = 0.001
ADAM_LEARNING_RATE_POLAR_DISTANCE = 0.001
ADAM_LEARNING_RATE_POS_AND_ANGLE = 0.001
ADAM_LEARNING_RATE_ANGLE_TO_TARGET = 0.0005
ADAM_LEARNING_RATE_LOOKING_ANGLE = 0.0005
ADAM_LEARNING_RATE_C = 0.0001

LAMBDA_L2 = 0
LAMBDA_L2_POLAR_ANGLE = 0
LAMBDA_L2_POLAR_DISTANCE = 0
LAMBDA_L2_POS_AND_ANGLE = 0
LAMBDA_L2_ANGLE_TO_TARGET = 1000
LAMBDA_L2_LOOKING_ANGLE = 100
LAMBDA_L2_C = 0

EMBEDDINGS_FILE = 'data/embedings.txt'
LABELS_FILE = 'data/labels.txt'
X_COORD_FILE = 'data/x_file.txt'
Y_COORD_FILE = 'data/y_file.txt'
ROTATION_FILE = 'data/rotation_file.txt'
SPEC_LAYER_FILE = 'data/spec_layer.txt'
TARGET_FILE = 'data/target.txt'
TARGET_X_FILE = 'data/target_x.txt'
TARGET_Y_FILE = 'data/target_y.txt'
STATE_ID_FILE = 'data/state_id.txt'
TARGET_EQ_OBS_FILE = 'data/target_eq_obs.txt'

TRAINED = False

SEED = 1

#DECODER_DIR = 'decoder_all_targets_targ_eq_obs_theta'
DECODER_DIR = 'decoder_all_targets_xy'

DECODER_CHECKPOINT_DIR = 'checkpoints/' + DECODER_DIR + '/'

# decoder2 directory contains logs of classifier

DECODER_LOG_FILE = 'logs/' + DECODER_DIR
