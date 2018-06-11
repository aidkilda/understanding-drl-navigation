# -*- coding: utf-8 -*-
import tensorflow as tf
from internal_representation_analysis.constants import ADAM_LEARNING_RATE
from internal_representation_analysis.constants import LAMBDA_L2

from internal_representation_analysis.decoder.models.DecoderModel import DecoderModel
from internal_representation_analysis.decoder.utils.decoder_network_utils import define_scope
from internal_representation_analysis.decoder.utils.decoder_network_utils import fc_bias_variable
from internal_representation_analysis.decoder.utils.decoder_network_utils import fc_weight_variable


# Regressor network for postion decoding
class PositionDecoderRegressor(DecoderModel):
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
        super(PositionDecoderRegressor, self).__init__(
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
        self.W_fc2_x = None
        self.W_fc2_y = None

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

        # Linear layer to predict x coordinates
        with tf.name_scope('W_fc2_x'):
            W_fc2_x = fc_weight_variable([data_size, 1])
            self.W_fc2_x = W_fc2_x

        with tf.name_scope('bias_fc2_x'):
            bias_fc2_x = fc_bias_variable([1], data_size)

        with tf.name_scope('out_x'):
            out_x_ = tf.matmul(h_fc1_drop, W_fc2_x) + bias_fc2_x
            out_x = tf.reshape(out_x_, [-1])

        # Linear layer to predict y coordinates
        with tf.name_scope('W_fc2_y'):
            W_fc2_y = fc_weight_variable([data_size, 1])
            self.W_fc2_y = W_fc2_y
        with tf.name_scope('bias_fc2_y'):
            bias_fc2_y = fc_bias_variable([1], data_size)

        with tf.name_scope('out_y'):
            out_y_ = tf.matmul(h_fc1_drop, W_fc2_y) + bias_fc2_y
            out_y = tf.reshape(out_y_, [-1])

        return out_x, out_y

    @define_scope
    def loss(self):

        with tf.name_scope('regularization_x'):
            regularization_x = LAMBDA_L2 * (
                tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2_x))

        with tf.name_scope('regularization_y'):
            regularization_y = LAMBDA_L2 * (
                tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2_y))

        pred_x, pred_y = self.prediction

        with tf.name_scope("loss_x"):
            mse_loss_with_L2_regularization_x = tf.divide(tf.reduce_sum(
                tf.squared_difference(x=self.x, y=pred_x)) + regularization_x, self.input_size)

        with tf.name_scope("loss_y"):
            mse_loss_with_L2_regularization_y = tf.divide(tf.reduce_sum(
                tf.squared_difference(x=self.y, y=pred_y)) + regularization_y, self.input_size)

        return mse_loss_with_L2_regularization_x, mse_loss_with_L2_regularization_y

    @define_scope
    def optimize(self):
        x_loss, y_loss = self.loss
        optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_LEARNING_RATE)
        with tf.name_scope("optimize_x"):
            opt_x = optimizer.minimize(x_loss)

        with tf.name_scope("optimize_y"):
            opt_y = optimizer.minimize(y_loss)

        return opt_x, opt_y

    @define_scope
    def summary(self):
        x_loss, y_loss = self.loss
        mean_err_x, var_err_x, mean_err_y, var_err_y = self.accuracy
        scalar_summaries = [
            tf.summary.scalar(self.scene_scope + "/x_loss", x_loss),
            tf.summary.scalar(self.scene_scope + "/y_loss", y_loss),
            tf.summary.scalar(self.scene_scope + "/x_mean", mean_err_x),
            tf.summary.scalar(self.scene_scope + "/y_mean", mean_err_y)
        ]
        return tf.summary.merge(scalar_summaries)

    @define_scope
    def accuracy(self):
        pred_x, pred_y = self.prediction
        error_x = tf.abs(pred_x - self.x)
        error_y = tf.abs(pred_y - self.y)

        mean_err_x, var_err_x = tf.nn.moments(error_x, axes=[0])
        mean_err_y, var_err_y = tf.nn.moments(error_y, axes=[0])
        return mean_err_x, var_err_x, mean_err_y, var_err_y
