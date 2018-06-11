# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class ActorCriticNetwork(object):
    """Actor-Critic Network Base Class.
    The policy network and value network architecture should be implemented in a child class of this one.
    """
    def __init__(self, action_size, device="/cpu:0", scope="network"):
        self._device = device
        self._action_size = action_size
        self.scope = scope

    def prepare_loss(self, entropy_beta):

        with tf.device(self._device) and tf.variable_scope("actor_critic_prepare_loss"):

            with tf.variable_scope("policy_loss"):
                # taken action (input for policy)
                self.a = tf.placeholder("float", [None, self._action_size], name="action_taken")

                # temporary difference (R-V) (input for policy)
                self.td = tf.placeholder("float", [None], name="td")

                # avoid NaN with clipping when value in pi becomes zero
                log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0, name="clip_by_value_pi"), name="log_pi")

                # policy entropy
                entropy = -tf.reduce_sum(self.pi * log_pi, axis=1, name="entropy")

                # policy loss (output)
                self.policy_loss = -tf.reduce_sum(
                    tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td +
                    entropy * entropy_beta, name="policy_loss")

            with tf.variable_scope("value_loss"):
                # R (input for value)
                self.r = tf.placeholder("float", [None], name="reward_value")

                # value loss (output)
                # learning rate for critic is half of actor's
                self.value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v, name="value_loss")

            with tf.variable_scope("total_loss"):
                # gradienet of policy and value are summed up
                self.total_loss = tf.add(self.policy_loss, self.value_loss, name="total_loss")

    def run_policy_and_value(self, sess, state, target):
        raise NotImplementedError()

    def run_policy(self, sess, state, target):
        raise NotImplementedError()

    def run_value(self, sess, state, target):
        raise NotImplementedError()

    def sync_from(self, src_network_scope):

        with tf.device(self._device):
            with tf.name_scope("sync"):
                src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, src_network_scope)
                dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

                sync_ops = []
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops)

    # weight initialization based on muupan's code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_variable(self, weight_shape, name=None):
        with tf.variable_scope("init_weights_biases"):
            input_channels = weight_shape[0]
            output_channels = weight_shape[1]
            d = 1.0 / np.sqrt(input_channels)
            bias_shape = [output_channels]
            weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name="weight_"+name)
            bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d), name="bias_"+name)
            return weight, bias