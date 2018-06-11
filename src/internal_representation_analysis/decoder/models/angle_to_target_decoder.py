# -*- coding: utf-8 -*-
import tensorflow as tf
from internal_representation_analysis.constants import ADAM_LEARNING_RATE_ANGLE_TO_TARGET
from internal_representation_analysis.constants import LAMBDA_L2_ANGLE_TO_TARGET

from internal_representation_analysis.decoder.models.DecoderModel import DecoderModel
from internal_representation_analysis.decoder.utils.decoder_network_utils import define_scope
from internal_representation_analysis.decoder.utils.decoder_network_utils import fc_bias_variable
from internal_representation_analysis.decoder.utils.decoder_network_utils import fc_weight_variable


# Regressor network for postion decoding
class AngleToTargetDecoder(DecoderModel):
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
        super(AngleToTargetDecoder, self).__init__(
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
        self.W_fc2 = None
        self.W_fc3 = None

    @define_scope
    def prediction(self):
        data_size = int(self.embedding.get_shape()[1])

        # Linear layer to predict angle
        with tf.name_scope('W_fc1'):
            W_fc1 = fc_weight_variable([data_size, data_size])
            self.W_fc1 = W_fc1

        with tf.name_scope('bias_fc1'):
            bias_fc1 = fc_bias_variable([data_size], data_size)

        with tf.name_scope('h_fc1'):
            h_fc1 = tf.nn.relu(tf.matmul(self.embedding, W_fc1) + bias_fc1)

        with tf.name_scope('dropout_h_fc1'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        # Linear layer to predict angle
        with tf.name_scope('W_fc2'):
            W_fc2 = fc_weight_variable([data_size, data_size])
            self.W_fc2 = W_fc2

        with tf.name_scope('bias_fc2'):
            bias_fc2 = fc_bias_variable([data_size], data_size)

        with tf.name_scope('h_fc2'):
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + bias_fc2)

        with tf.name_scope('dropout_h_fc2'):
            h_fc2_drop = tf.nn.dropout(h_fc2, self.dropout_keep_prob)

        # Linear layer to predict angle
        with tf.name_scope('W_fc3'):
            W_fc3 = fc_weight_variable([data_size, 1])
            self.W_fc3 = W_fc3

        with tf.name_scope('bias_fc3'):
            bias_fc3 = fc_bias_variable([1], data_size)

        with tf.name_scope('out_angle'):
            out_angle_ = tf.matmul(h_fc2_drop, W_fc3) + bias_fc3
            out_angle = tf.reshape(out_angle_, [-1])

        return out_angle

    @define_scope
    def loss(self):

        with tf.name_scope('regularization_angle'):
            regularization_angle = LAMBDA_L2_ANGLE_TO_TARGET * (
                tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2) + tf.nn.l2_loss(self.W_fc3))

        pred_angle = tf.mod(self.prediction, 360)

        with tf.name_scope("loss_angle"):
            anticlockwise_err = tf.abs(pred_angle - self.target_angle)
            clockwise_err = 360.0 - anticlockwise_err
            error_angle = tf.minimum(anticlockwise_err, clockwise_err)
            mse_loss_with_L2_regularization_angle = tf.divide(tf.reduce_sum(tf.square(
                error_angle)) + regularization_angle, self.input_size)



        return mse_loss_with_L2_regularization_angle

    @define_scope
    def optimize(self):
        angle_loss = self.loss
        optimizer_angle = tf.train.AdamOptimizer(learning_rate=ADAM_LEARNING_RATE_ANGLE_TO_TARGET)
        with tf.name_scope("optimize_angle"):
            opt_angle = optimizer_angle.minimize(angle_loss)

        return opt_angle

    @define_scope
    def summary(self):
        angle_loss = self.loss
        mean_err_angle, var_err_angle = self.accuracy
        scalar_summaries = [
            tf.summary.scalar(self.scene_scope + "/angle_loss", angle_loss),
            tf.summary.scalar(self.scene_scope + "/angle_mean", mean_err_angle),
            tf.summary.scalar(self.scene_scope + "/angle_var", var_err_angle)
        ]
        return tf.summary.merge(scalar_summaries)

    @define_scope
    def accuracy(self):
        pred_angle = self.prediction
        anticlockwise_err = tf.abs(pred_angle - self.target_angle)
        clockwise_err = 360.0 - anticlockwise_err
        error_angle = tf.minimum(anticlockwise_err, clockwise_err)

        mean_err_angle, var_err_angle = tf.nn.moments(error_angle, axes=[0])
        return mean_err_angle, var_err_angle
