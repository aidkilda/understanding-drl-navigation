# -*- coding: utf-8 -*-
import tensorflow as tf
from internal_representation_analysis.constants import ADAM_LEARNING_RATE_POLAR_ANGLE
from internal_representation_analysis.constants import ADAM_LEARNING_RATE_POLAR_DISTANCE
from internal_representation_analysis.constants import LAMBDA_L2_POLAR_ANGLE
from internal_representation_analysis.constants import LAMBDA_L2_POLAR_DISTANCE

from internal_representation_analysis.decoder.models.DecoderModel import DecoderModel
from internal_representation_analysis.decoder.utils.decoder_network_utils import define_scope
from internal_representation_analysis.decoder.utils.decoder_network_utils import fc_bias_variable
from internal_representation_analysis.decoder.utils.decoder_network_utils import fc_weight_variable


# Regressor network for postion decoding
class PositionPolarDecoderRegressor(DecoderModel):
    def __init__(self,
                 scene_scope,
                 embedding,
                 x,
                 y,
                 theta,
                 label,
                 target,
                 angle,
                 r,
                 target_angle,
                 target_distance,
                 dropout_keep_prob,
                 input_size,
                 device="/cpu:0"):
        super(PositionPolarDecoderRegressor, self).__init__(
            scene_scope=scene_scope,
            embedding=embedding,
            x=x,
            y=y,
            theta=theta,
            label=label,
            target=target,
            angle=angle,
            r=r,
            target_angle=target_angle,
            target_distance=target_distance,
            dropout_keep_prob=dropout_keep_prob,
            input_size=input_size,
            device=device)
        self.W_fc1 = None
        self.W_fc2_angle = None
        self.W_fc2_r = None

    @define_scope
    def prediction(self):
        data_size = int(self.embedding.get_shape()[1])

        # Shared hidden layer
        with tf.name_scope('W_shared_fc1'):
            W_fc1 = fc_weight_variable([data_size, data_size])
            self.W_fc1 = W_fc1

        with tf.name_scope('bias_shared_fc1'):
            bias_fc1 = fc_bias_variable([data_size], data_size)

        with tf.name_scope('h_shared_fc1'):
            h_fc1 = tf.nn.relu(tf.matmul(self.embedding, W_fc1) + bias_fc1)

        with tf.name_scope('dropout_h_fc1'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        # Linear layer to predict angle
        with tf.name_scope('W_fc2_angle'):
            W_fc2_angle = fc_weight_variable([data_size, 1])
            self.W_fc2_angle = W_fc2_angle

        with tf.name_scope('bias_fc2_angle'):
            bias_fc2_angle = fc_bias_variable([1], data_size)

        with tf.name_scope('out_angle'):
            out_angle_ = tf.matmul(h_fc1_drop, W_fc2_angle) + bias_fc2_angle
            out_angle = tf.reshape(out_angle_, [-1])

        # Linear layer to predict distance
        with tf.name_scope('W_fc2_r'):
            W_fc2_r = fc_weight_variable([data_size, 1])
            self.W_fc2_r = W_fc2_r
        with tf.name_scope('bias_fc2_r'):
            bias_fc2_r = fc_bias_variable([1], data_size)

        with tf.name_scope('out_r'):
            out_r_ = tf.matmul(h_fc1_drop, W_fc2_r) + bias_fc2_r
            out_r = tf.reshape(out_r_, [-1])

        return out_angle, out_r

    @define_scope
    def loss(self):

        with tf.name_scope('regularization_angle'):
            regularization_angle = LAMBDA_L2_POLAR_ANGLE * (
                tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2_angle))

        with tf.name_scope('regularization_r'):
            regularization_r = LAMBDA_L2_POLAR_DISTANCE * (
                tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2_r))

        pred_angle, pred_r = self.prediction

        with tf.name_scope("loss_angle"):
            mse_loss_with_L2_regularization_angle = tf.divide(tf.reduce_sum(
                tf.squared_difference(x=self.angle,
                                      y=pred_angle)) + regularization_angle, self.input_size)

        with tf.name_scope("loss_r"):
            mse_loss_with_L2_regularization_r = tf.divide(tf.reduce_sum(
                tf.squared_difference(x=self.r, y=pred_r)) + regularization_r, self.input_size)

        return mse_loss_with_L2_regularization_angle, mse_loss_with_L2_regularization_r

    @define_scope
    def optimize(self):
        angle_loss, r_loss = self.loss
        optimizer_angle = tf.train.AdamOptimizer(learning_rate=ADAM_LEARNING_RATE_POLAR_ANGLE)
        optimizer_distance = tf.train.AdamOptimizer(learning_rate=ADAM_LEARNING_RATE_POLAR_DISTANCE)
        with tf.name_scope("optimize_angle"):
            opt_angle = optimizer_angle.minimize(angle_loss)

        with tf.name_scope("optimize_distance"):
            opt_distance = optimizer_distance.minimize(r_loss)

        return opt_angle, opt_distance

    @define_scope
    def summary(self):
        angle_loss, r_loss = self.loss
        mean_err_angle, var_err_angle, mean_err_r, var_err_r = self.accuracy
        scalar_summaries = [
            tf.summary.scalar(self.scene_scope + "/angle_loss", angle_loss),
            tf.summary.scalar(self.scene_scope + "/r_loss", r_loss),
            tf.summary.scalar(self.scene_scope + "/angle_mean", mean_err_angle),
            tf.summary.scalar(self.scene_scope + "/r_mean", mean_err_r),
        ]
        return tf.summary.merge(scalar_summaries)

    @define_scope
    def accuracy(self):
        pred_angle, pred_r = self.prediction
        error_angle = tf.abs(pred_angle - self.angle)
        error_r = tf.abs(pred_r - self.r)

        mean_err_angle, var_err_angle = tf.nn.moments(error_angle, axes=[0])
        mean_err_r, var_err_r = tf.nn.moments(error_r, axes=[0])
        return mean_err_angle, var_err_angle, mean_err_r, var_err_r