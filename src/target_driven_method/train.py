import tensorflow as tf
import threading
import datetime

import signal
import os

from target_driven_method.networks.A3C_networks import A3CFFNetwork
from target_driven_method.networks.target_driven_navigation_networks import TargetDrivenFFNetwork
from target_driven_method.networks.A3C_networks import A3CLSTMNetwork
from target_driven_method.networks.target_driven_navigation_networks import TargetDrivenLSTMNetwork
from training_thread import A3CTrainingThread
from utils.ops import log_uniform
from Grid2DEnvAdapterForTargetDriven import Grid2DEnvAdapter

from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import USE_GPU
from constants import USE_LSTM

from constants import config

if __name__ == '__main__':

    device = "/gpu:0" if USE_GPU else "/cpu:0"

    global_t = 0
    stop_requested = False

    unique_name = "{}__{}".format(config['name'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    EXPERIMENT_CHECKPOINT_DIR = CHECKPOINT_DIR + "/" + unique_name
    EXPERIMENT_LOG_DIR = LOG_DIR + "/" + unique_name

    if not os.path.exists(EXPERIMENT_CHECKPOINT_DIR):
        os.makedirs(EXPERIMENT_CHECKPOINT_DIR)

    if not os.path.exists(EXPERIMENT_LOG_DIR):
        os.makedirs(EXPERIMENT_LOG_DIR)

    initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH,
                                        INITIAL_ALPHA_LOG_RATE)

    env = Grid2DEnvAdapter(config)

    global_network_scope = "global"

    if config['model'] == "TDN":
        if USE_LSTM:
            global_network = TargetDrivenLSTMNetwork(
            input_size=env.obs_size,
            device=device,
            network_scope=global_network_scope)
        else:
            global_network = TargetDrivenFFNetwork(
                input_size=env.obs_size,
                device=device,
                network_scope=global_network_scope)
    else:
        if USE_LSTM:
            global_network = A3CLSTMNetwork(
            input_size=env.obs_size,
            device=device,
            network_scope=global_network_scope)
        else:
            global_network = A3CFFNetwork(
                input_size=env.obs_size,
                device=device,
                network_scope=global_network_scope)

    learning_rate_input = tf.placeholder("float")
    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=RMSP_ALPHA,
        momentum=0.0,
        epsilon=RMSP_EPSILON)

    # instantiate each training thread
    training_threads = []

    for i in range(PARALLEL_SIZE):
        training_thread = A3CTrainingThread(
            config,
            env,
            i,
            global_network_scope,
            initial_learning_rate,
            learning_rate_input,
            grad_applier,
            MAX_TIME_STEP,
            device=device
        )
        training_threads.append(training_thread)

    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True))

    init = tf.global_variables_initializer()
    sess.run(init)

    # create tensorboard summaries
    summary_op = dict()
    summary_placeholders = dict()

    for i in range(PARALLEL_SIZE):
        key = unique_name + "_thread_" + str(i)

        # summary for tensorboard
        episode_reward_input = tf.placeholder("float")
        episode_length_input = tf.placeholder("float")
        episode_mean_loss_input = tf.placeholder("float")

        scalar_summaries = [
            tf.summary.scalar(key + "/Episode Reward", episode_reward_input),
            tf.summary.scalar(key + "/Episode Length", episode_length_input),
            tf.summary.scalar(key + "/Episode Mean Loss", episode_mean_loss_input),
        ]

        summary_op[key] = tf.summary.merge(scalar_summaries)
        summary_placeholders[key] = {
            "episode_reward_input": episode_reward_input,
            "episode_length_input": episode_length_input,
            "episode_mean_loss_input": episode_mean_loss_input,
            "learning_rate_input": learning_rate_input
        }

    summary_writer = tf.summary.FileWriter(EXPERIMENT_LOG_DIR, sess.graph)

    # init or load checkpoint with saver
    # if you don't need to be able to resume training, use the next line instead.
    # it will result in a much smaller checkpoint file.
    # saver = tf.train.Saver(max_to_keep=10, var_list=global_network.get_vars())
    saver = tf.train.Saver(max_to_keep=1000)

    def train_function(parallel_index):
        global global_t
        training_thread = training_threads[parallel_index]
        last_global_t = 0

        key = unique_name + "_thread_" + str(parallel_index)

        while global_t < MAX_TIME_STEP and not stop_requested:
            diff_global_t = training_thread.process(
                sess, global_t, summary_writer, summary_op[key],
                summary_placeholders[key])
            global_t += diff_global_t
            # periodically save checkpoints to disk and test performance
            if parallel_index == 0 and global_t - last_global_t > 10000:
                print('Save checkpoint at timestamp %d' % global_t)
                saver.save(
                    sess,
                    EXPERIMENT_CHECKPOINT_DIR + '/' + 'checkpoint',
                    global_step=global_t)
                last_global_t = global_t

    def signal_handler(signal, frame):
        global stop_requested
        print('You pressed Ctrl+C!')
        stop_requested = True

    train_threads = []
    for i in range(PARALLEL_SIZE):
        train_threads.append(
            threading.Thread(target=train_function, args=(i, )))

    signal.signal(signal.SIGINT, signal_handler)

    # start each training thread
    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop.')
    signal.pause()

    # wait for all threads to finish
    for t in train_threads:
        t.join()

    print('Now saving data. Please wait.')
    saver.save(sess, EXPERIMENT_CHECKPOINT_DIR + '/' + 'checkpoint', global_step=global_t)
    summary_writer.close()
