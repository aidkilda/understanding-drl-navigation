import tensorflow as tf
import numpy as np

from constants import ACTION_SIZE
from constants import HISTORY_LENGTH
from constants import HIDDEN_NEURONS

from target_driven_method.networks.network import ActorCriticNetwork

class TargetDrivenFFNetwork(ActorCriticNetwork):
    """Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017],
    without scene-specific layers.
    """
    def __init__(self,
                 input_size,
                 device="/cpu:0",
                 network_scope="network"):
        ActorCriticNetwork.__init__(self, ACTION_SIZE, device, network_scope)

        with tf.device(self._device):

            with tf.variable_scope(network_scope):

                with tf.variable_scope("input_layer"):
                    # state (input)
                    self.s = tf.placeholder("float", [None, input_size, HISTORY_LENGTH], name="state")

                    # target (input)
                    self.t = tf.placeholder("float", [None, input_size, HISTORY_LENGTH], name="target")

                    # flatten input
                    self.s_flat = tf.reshape(self.s, [-1, input_size * HISTORY_LENGTH], name="state_flat")
                    self.t_flat = tf.reshape(self.t, [-1, input_size * HISTORY_LENGTH], name="target_flat")

                with tf.variable_scope("shared_siamese_layer"):
                    self.W_fc1, self.b_fc1 = \
                        self._fc_variable([input_size * HISTORY_LENGTH, HIDDEN_NEURONS], name="shared_siamese")

                    h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1) + self.b_fc1,
                                          name="shared_siamese_state_flat")
                    h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1) + self.b_fc1,
                                          name="shared_siamese_target_flat")
                    h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1, name="h_shared_siamese")

                    self.observation_embedding = h_s_flat

                with tf.variable_scope("shared_fusion_layer"):
                    self.W_fc2, self.b_fc2 = \
                        self._fc_variable([2 * HIDDEN_NEURONS, HIDDEN_NEURONS], name="shared_fusion")
                    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2, name="h_shared_fusion")

                with tf.variable_scope("fc_layer_3"):
                    self.W_fc3, self.b_fc3 = self._fc_variable([HIDDEN_NEURONS,HIDDEN_NEURONS], name="fc3")
                    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3, name="h_fc3")

                    self.scene_specific_layer = h_fc3

                with tf.variable_scope("policy_output_layer"):
                    self.W_policy, self.b_policy = self._fc_variable([HIDDEN_NEURONS, ACTION_SIZE], name="policy")

                    # policy (output)
                    pi_ = tf.matmul(h_fc3, self.W_policy) + self.b_policy
                    self.pi_ = pi_
                    self.pi = tf.nn.softmax(pi_, name="pi")

                with tf.variable_scope("value_output_layer"):
                    self.W_value, self.b_value = self._fc_variable([HIDDEN_NEURONS, 1], name="value")

                    # value (output)
                    v_ = tf.matmul(h_fc3, self.W_value) + self.b_value
                    self.v = tf.reshape(v_, [-1], name="value")

    def run_policy_and_value(self, sess, state, target):
        pi_out, v_out = sess.run(
            [self.pi, self.v],
            feed_dict={
                self.s: [state],
                self.t: [target]
            })
        return pi_out[0], v_out[0]

    def run_policy(self, sess, state, target):
        pi_out = sess.run(
            self.pi, feed_dict={
                self.s: [state],
                self.t: [target]
            })
        return pi_out[0]

    def run_value(self, sess, state, target):
        v_out = sess.run(
            self.v, feed_dict={
                self.s: [state],
                self.t: [target]
            })
        return v_out[0]


class TargetDrivenLSTMNetwork(ActorCriticNetwork):
    """Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017],
    without scene-specific layers and with added LSTM layer.
    """
    def __init__(self,
                 input_size,
                 device="/cpu:0",
                 network_scope="network"):
        ActorCriticNetwork.__init__(self, ACTION_SIZE, device, network_scope)

        with tf.device(self._device):

            with tf.variable_scope(network_scope) as scope:

                with tf.variable_scope("input_layer"):
                    # state (input)
                    self.s = tf.placeholder("float", [None, input_size, HISTORY_LENGTH], name="state")

                    # target (input)
                    self.t = tf.placeholder("float", [None, input_size, HISTORY_LENGTH], name="target")

                    # flatten input
                    self.s_flat = tf.reshape(self.s, [-1, input_size * HISTORY_LENGTH], name="state_flat")
                    self.t_flat = tf.reshape(self.t, [-1, input_size * HISTORY_LENGTH], name="target_flat")

                with tf.variable_scope("shared_siamese_layer"):
                    self.W_fc1, self.b_fc1 = \
                        self._fc_variable([input_size * HISTORY_LENGTH, HIDDEN_NEURONS], name="shared_siamese")

                    h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1) + self.b_fc1,
                                          name="shared_siamese_state_flat")
                    h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1) + self.b_fc1,
                                          name="shared_siamese_target_flat")
                    h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1, name="h_shared_siamese")

                    self.observation_embedding = h_s_flat

                with tf.variable_scope("shared_fusion_layer"):
                    self.W_fc2, self.b_fc2 = \
                        self._fc_variable([2 * HIDDEN_NEURONS, HIDDEN_NEURONS], name="shared_fusion")
                    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2, name="h_shared_fusion")

                    h_fc_2_reshaped = tf.reshape(h_fc2, [1,-1,HIDDEN_NEURONS])

                with tf.variable_scope("lstm_layer"):
                    self.lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_NEURONS, state_is_tuple=True)
                    self.step_size = tf.placeholder(tf.float32, [1])
                    self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, HIDDEN_NEURONS])
                    self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, HIDDEN_NEURONS])
                    self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                            self.initial_lstm_state1)

                    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                                      h_fc_2_reshaped,
                                                                      initial_state=self.initial_lstm_state,
                                                                      sequence_length=self.step_size,
                                                                      time_major=False)

                    h_fc3 = tf.reshape(lstm_outputs, [-1,HIDDEN_NEURONS])

                with tf.variable_scope("policy_output_layer"):
                    self.W_policy, self.b_policy = self._fc_variable([HIDDEN_NEURONS, ACTION_SIZE], name="policy")

                    # policy (output)
                    pi_ = tf.matmul(h_fc3, self.W_policy) + self.b_policy
                    self.pi_ = pi_
                    self.pi = tf.nn.softmax(pi_, name="pi")

                with tf.variable_scope("value_output_layer"):
                    self.W_value, self.b_value = self._fc_variable([HIDDEN_NEURONS, 1], name="value")

                    # value (output)
                    v_ = tf.matmul(h_fc3, self.W_value) + self.b_value
                    self.v = tf.reshape(v_, [-1], name="value")

                self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, HIDDEN_NEURONS]),
                                                            np.zeros([1, HIDDEN_NEURONS]))

    def run_policy_and_value(self, sess, state, target):
        pi_out, v_out, self.lstm_state_out = sess.run(
            [self.pi, self.v, self.lstm_state],
            feed_dict={
                self.s: [state],
                self.t: [target],
                self.initial_lstm_state0: self.lstm_state_out[0],
                self.initial_lstm_state1: self.lstm_state_out[1],
                self.step_size: [1]
            })
        return pi_out[0], v_out[0]

    def run_policy(self, sess, state, target):
        pi_out, self.lstm_state_out = sess.run(
            [self.pi, self.lstm_state],
            feed_dict={
                self.s: [state],
                self.t: [target],
                self.initial_lstm_state0: self.lstm_state_out[0],
                self.initial_lstm_state1: self.lstm_state_out[1],
                self.step_size: [1]
            })
        return pi_out[0]

    def run_value(self, sess, state, target):
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run(
            [self.v, self.lstm_state],
            feed_dict={
                self.s: [state],
                self.t: [target],
                self.initial_lstm_state0: self.lstm_state_out[0],
                self.initial_lstm_state1: self.lstm_state_out[1],
                self.step_size: [1]
            })
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]