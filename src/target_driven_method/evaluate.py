import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from target_driven_method.networks.A3C_networks import A3CFFNetwork
from target_driven_method.networks.target_driven_navigation_networks import TargetDrivenFFNetwork
from target_driven_method.networks.A3C_networks import A3CLSTMNetwork
from target_driven_method.networks.target_driven_navigation_networks import TargetDrivenLSTMNetwork

from Grid2DEnvAdapterForTargetDriven import Grid2DEnvAdapter

from constants import NUM_EVAL_EPISODES
from constants import SEED
from constants import USE_LSTM
from constants import config,task_list
from utils.ops import sample_action

def plot_learning_curves(task_results):
    # Create means and standard deviations of training set scores
    train_mean = np.asarray(task_results['mean'])
    train_std = np.asarray(task_results['std'])
    train_steps = np.asarray(task_results['step'])

    # Draw lines
    plt.plot(train_steps, train_mean, color="#000080", label="Path length")

    # Draw bands
    plt.fill_between(train_steps, train_mean - train_std, train_mean + train_std, color="#99cc00")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Step"), plt.ylabel("Path Length")
    plt.tight_layout()

    save_path = 'cog_res/metric/BA/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'lc.pdf')
    plt.savefig(save_path + 'lc.png')
    plt.close()

if __name__ == '__main__':

    np.random.seed(SEED)
    random.seed(SEED)
    tf.set_random_seed(SEED)

    device = "/cpu:0"

    target_img = None
    if config['task_list']:
        task = task_list[1]
        config['initial_agent'] = task[0]
        config['initial_target'] = task[1]
        config['target_value'] = task[2]
        target_img_id = task[3]
        print("Target id:", task[3])

    env = Grid2DEnvAdapter(config)

    if config['task_list']:
        target_img = env.tiled_state(target_img_id)

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

    # Insert a unique name of the model that has to be evaluated.
    unique_name = "metric__2018-05-03_18-55-37"
    EXPERIMENT_CHECKPOINT_DIR = "/home/aidas/Masters/understanding-drl-navigation/src/target_driven_method/checkpoints"\
                                + "/" + unique_name

    checkpoint = tf.train.get_checkpoint_state(EXPERIMENT_CHECKPOINT_DIR)

    task_results = {}
    task_results['mean'] = []
    task_results['std'] = []
    task_results['step'] = []

    paths = checkpoint.all_model_checkpoint_paths

    for model_path in checkpoint.all_model_checkpoint_paths:

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        saver.restore(sess, model_path)
        print("checkpoint loaded: {}".format(model_path))

        step = int(model_path.split("-")[-1])
        print("Step", step)

        ep_rewards = []
        ep_lengths = []
        ep_lengths_below_100 = []
        ep_shortest_distances = []
        ep_longer_than_100 = 0
        trajectories = []

        for i_episode in range(NUM_EVAL_EPISODES):

            env.reset()

            if USE_LSTM:
                global_network.reset_state()

            agent_x = env.x
            agent_y = env.y
            target_x = env.get_x(env.terminal_state_id)
            target_y = env.get_y(env.terminal_state_id)
            shortest_distance = np.abs(agent_x - target_x) + np.abs(agent_y - target_y) + 2

            terminal = False
            ep_reward = 0
            ep_t = 0
            list_env_id = []

            trajectory = []

            while not terminal:

                list_env_id.append(env.current_state_id)

                trajectory.append((env.x, env.y))

                pi_values = global_network.run_policy(
                    sess, env.s_t, target_img)
                action = sample_action(pi_values)
                env.step(action)

                terminal = env.terminal
                if ep_t == 100:
                    ep_longer_than_100 += 1
                    break
                ep_reward += env.reward
                ep_t += 1
            if ep_t < 100:
                ep_lengths.append(ep_t)
                ep_rewards.append(ep_reward)
                ep_shortest_distances.append(shortest_distance)
                trajectories.append(trajectory)
            ep_lengths_below_100.append(ep_t)

        mean = np.mean(ep_lengths_below_100)
        std = np.std(ep_lengths_below_100)

        task_results['mean'].append(mean)
        task_results['std'].append(std)
        task_results['step'].append(step)

        print('\nResults (mean trajectory length, std trajectory length, shortest trajectory length, number of episodes shorter than 100):')
        print('%.2f steps' % (np.mean(ep_lengths)))
        print('%.2f steps' % (np.std(ep_lengths)))
        print('%.2f steps' % (np.mean(ep_shortest_distances)))
        print('%.2f episodes' % (NUM_EVAL_EPISODES - ep_longer_than_100))

    plot_learning_curves(task_results)

