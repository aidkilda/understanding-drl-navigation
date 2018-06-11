import tensorflow as tf
import numpy as np
import sys

from Grid2DEnvAdapterForTargetDriven import Grid2DEnvAdapter
from target_driven_method.networks.A3C_networks import A3CFFNetwork
from target_driven_method.networks.target_driven_navigation_networks import TargetDrivenFFNetwork
from target_driven_method.networks.A3C_networks import A3CLSTMNetwork
from target_driven_method.networks.target_driven_navigation_networks import TargetDrivenLSTMNetwork
from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import GRAD_NORM_CLIP
from constants import USE_LSTM


class A3CTrainingThread(object):
    def __init__(self,
                 config,
                 env,
                 thread_index,
                 global_network_scope,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.target_img_id = None
        if config['task_list']:
            tasks = config['task_list']
            task = tasks[thread_index % len(tasks)]
            config['initial_agent'] = task[0]
            config['initial_target'] = task[1]
            config['target_value'] = task[2]
            self.target_img_id = task[3]

        self.env = Grid2DEnvAdapter(config)

        self.scope = "thread-%d" % (thread_index + 1)

        print("Navigation target", self.scope, self.env.terminal_state_id)

        if config['model'] == "TDN":
            if USE_LSTM:
                self.local_network = TargetDrivenLSTMNetwork(
                    input_size=self.env.obs_size,
                    device=device,
                    network_scope=self.scope)
            else:
                self.local_network = TargetDrivenFFNetwork(
                    input_size=self.env.obs_size,
                    device=device,
                    network_scope=self.scope)
        else:
            if USE_LSTM:
                self.local_network = A3CLSTMNetwork(
                    input_size=self.env.obs_size,
                    device=device,
                    network_scope=self.scope)
            else:
                self.local_network = A3CFFNetwork(
                    input_size=self.env.obs_size,
                    device=device,
                    network_scope=self.scope)

        self.local_network.prepare_loss(ENTROPY_BETA)

        with tf.name_scope("get_gradients-%d" % (thread_index + 1)):

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

            grads_and_vars = grad_applier.compute_gradients(self.local_network.total_loss, local_vars)
            grads = [grad_and_var[0] for grad_and_var in grads_and_vars]

            with tf.name_scope("clip_gradients-%d" % (thread_index + 1)):
                clipped_grads = [tf.clip_by_norm(grad, GRAD_NORM_CLIP) for grad in grads]

        with tf.name_scope("apply_gradients-%d" % (thread_index + 1)):
            self.apply_gradients = grad_applier.apply_gradients(zip(clipped_grads,global_vars))

        self.sync = self.local_network.sync_from(global_network_scope)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0
        self.episode_length = 0
        self.episode_losses = []

    def _anneal_learning_rate(self, global_time_step):
        time_step_to_go = max(self.max_global_time_step - global_time_step,
                              0.0)
        learning_rate = self.initial_learning_rate * time_step_to_go / self.max_global_time_step
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, writer, summary_op, placeholders, values,
                      global_t):
        feed_dict = {}
        for k in placeholders:
            feed_dict[placeholders[k]] = values[k]

        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        writer.add_summary(summary_str, global_t)
        writer.flush()

    def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):

        states = []
        actions = []
        rewards = []
        values = []
        targets = []

        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        if USE_LSTM:
            start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(LOCAL_T_MAX):
            target_img = None
            if self.target_img_id:
                target_img = self.env.tiled_state(self.target_img_id)
            pi_, value_ = self.local_network.run_policy_and_value(
                sess, self.env.s_t, target_img)

            action = self.choose_action(pi_)

            states.append(self.env.s_t)
            actions.append(action)
            values.append(value_)
            targets.append(target_img)

            # process game
            self.env.step(action)

            # receive game result
            reward = self.env.reward
            terminal = self.env.terminal

            # clip reward
            clipped_reward = np.clip(reward, -1, 1)
            rewards.append(clipped_reward)

            R = 0.0
            if not terminal:
                R = self.local_network.run_value(sess, self.env.s_t, target_img)

            R = clipped_reward + GAMMA * R
            td = R - value_
            a = np.zeros([ACTION_SIZE])
            a[action] = 1

            if USE_LSTM:
                # Not implemented.
                loss_ = 0
            else:
                loss_ = sess.run(
                    self.local_network.total_loss,
                    feed_dict={
                        self.local_network.s: [self.env.s_t],
                        self.local_network.t: [target_img],
                        self.local_network.a: [a],
                        self.local_network.td: [td],
                        self.local_network.r: [R]})

            self.episode_reward += reward
            self.episode_length += 1

            self.episode_losses.append(loss_)

            self.local_t += 1

            if terminal:
                terminal_end = True
                episode_mean_loss = np.asarray(self.episode_losses).mean()
                sys.stdout.write(
                    "time %d | thread #%d | target %d | start %d |episode reward = %.3f | episode length = %d |"
                    "episode mean loss  = %.3f\n"
                    % (global_t,
                       self.thread_index,
                       self.env.terminal_state_id,
                       self.env.start_state_id,
                       self.episode_reward,
                       self.episode_length,
                       episode_mean_loss,))

                summary_values = {
                    "episode_reward_input": self.episode_reward,
                    "episode_length_input": float(self.episode_length),
                    "episode_mean_loss_input": episode_mean_loss,
                    "learning_rate_input": self._anneal_learning_rate(global_t)
                }

                self._record_score(sess, summary_writer, summary_op,
                                   summary_placeholders, summary_values,
                                   global_t)
                self.episode_reward = 0
                self.episode_length = 0
                self.episode_losses = []
                self.env.reset()

                if USE_LSTM:
                    self.local_network.reset_state()

                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.env.s_t, target_img)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []
        batch_t = []

        # compute and accumulate gradients
        for (ai, ri, si, Vi, ti) in zip(actions, rewards, states, values, targets):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)
            batch_t.append(ti)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        if USE_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()
            batch_t.reverse()

            sess.run(
                self.apply_gradients,
                feed_dict={
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.t: batch_t,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.local_network.initial_lstm_state: start_lstm_state,
                    self.local_network.step_size: [len(batch_a)],
                    self.learning_rate_input: cur_learning_rate})

        else:
            sess.run(
                self.apply_gradients,
                feed_dict={
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.t: batch_t,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.learning_rate_input: cur_learning_rate})

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
