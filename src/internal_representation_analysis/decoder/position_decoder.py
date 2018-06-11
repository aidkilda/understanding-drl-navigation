# -*- coding: utf-8 -*-
import sys
import os

import tensorflow as tf
from internal_representation_analysis.constants import DECODER_CHECKPOINT_DIR
from internal_representation_analysis.constants import DECODER_LOG_FILE
from internal_representation_analysis.constants import DECODER_DIR
from internal_representation_analysis.constants import DROPOUT_KEEP_PROB
from internal_representation_analysis.constants import DROPOUT_KEEP_PROB_POLAR_ANGLE
from internal_representation_analysis.constants import DROPOUT_KEEP_PROB_POS_AND_ANGLE
from internal_representation_analysis.constants import DROPOUT_KEEP_PROB_ANGLE_TO_TARGET
from internal_representation_analysis.constants import DROPOUT_KEEP_PROB_LOOKING_ANGLE
from internal_representation_analysis.constants import EPOCHS
from internal_representation_analysis.constants import MINI_BATCH_SIZE
from internal_representation_analysis.constants import VERBOSE

from internal_representation_analysis.decoder.models.position_decoder_regressor_network import PositionDecoderRegressor
from internal_representation_analysis.decoder.models.polar_coordinates_regressor_network import PositionPolarDecoderRegressor
from internal_representation_analysis.decoder.models.angle_to_target_decoder import AngleToTargetDecoder
from internal_representation_analysis.decoder.models.looking_angle_decoder import LookingAngleDecoder

class PositionDecoderTrainer(object):
    def __init__(self, scene_scope, state_dataset):
        self.scene_scope = scene_scope
        self.state_dataset = state_dataset
        self.training_data_size = len(self.state_dataset.train_set)

    def train(self, include_test):

        model = self.create_model()

        saver = tf.train.Saver()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        DECODER_LOG_FILE_TRAIN = DECODER_LOG_FILE + '/' + self.scene_scope + '/train'
        DECODER_LOG_FILE_VALIDATION = DECODER_LOG_FILE + '/' + self.scene_scope + '/val'
        DECODER_LOG_FILE_TEST = DECODER_LOG_FILE + '/' + self.scene_scope + '/test'
        train_summary_writer = tf.summary.FileWriter(DECODER_LOG_FILE_TRAIN,
                                                     sess.graph)
        val_summary_writer = tf.summary.FileWriter(DECODER_LOG_FILE_VALIDATION)
        test_summary_writer = tf.summary.FileWriter(DECODER_LOG_FILE_TEST)

        for epoch in range(EPOCHS):

            self.state_dataset.shuffle_train_set()

            for i in range(0, len(self.state_dataset.train_set),
                           MINI_BATCH_SIZE):

                states_mini = self.state_dataset.get_train_mini_batch(i)

                feed_dict = self.get_feed_dict(model, states_mini)
                feed_dict[model.dropout_keep_prob] = DROPOUT_KEEP_PROB
                _, summary = sess.run(
                    [model.optimize, model.summary], feed_dict=feed_dict)

            train_summary = sess.run(
                model.summary,
                feed_dict=self.get_feed_dict(
                    model, self.state_dataset.train_set))

            train_summary_writer.add_summary(train_summary,
                                           self.__global_step(epoch))

            val_summary = sess.run(
                model.summary,
                feed_dict=self.get_feed_dict(
                    model, self.state_dataset.validation_set))

            val_summary_writer.add_summary(val_summary,
                                           self.__global_step(epoch))

            if (include_test):
                test_summary = sess.run(
                    model.summary,
                    feed_dict=self.get_feed_dict(model,
                                                 self.state_dataset.test_set))

                test_summary_writer.add_summary(test_summary,
                                                self.__global_step(epoch))

        save_dir = DECODER_CHECKPOINT_DIR + '/' + self.scene_scope

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        saver.save(sess, save_dir + '/' + self.scene_scope)

        train_summary_writer.close()
        val_summary_writer.close()
        test_summary_writer.close()

    def get_feed_dict(self, model, dataset):
        return {
            model.x: [s.x for s in dataset],
            model.y: [s.y for s in dataset],
            model.theta: [s.theta for s in dataset],
            model.label: [s.label for s in dataset],
            model.embedding: [s.embedding for s in dataset],
            model.angle: [s.angle for s in dataset],
            model.r: [s.r for s in dataset],
            model.target_angle: [s.target_angle for s in dataset],
            model.target_distance: [s.target_distance for s in dataset],
            model.input_size: len(dataset)
        }

    def create_model(self):

        embedding_size = self.state_dataset.all_states[0].embedding.size
        label_size = len(self.state_dataset.all_states[0].label)

        x_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        y_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        theta_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        label_placeholder = tf.placeholder(tf.float32, [None, label_size])
        embedding_placeholder = tf.placeholder(tf.float32,
                                               [None, embedding_size])
        angle_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        r_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        target_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        target_angle_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        target_distance_placeholder = tf.placeholder(
            tf.float32, shape=[
                None,
            ])
        input_size_placeholder = tf.placeholder(
            tf.float32, shape=())

        dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

        model = PositionDecoderRegressor(
            scene_scope=self.scene_scope,
            embedding=embedding_placeholder,
            x=x_placeholder,
            y=y_placeholder,
            theta=theta_placeholder,
            label=label_placeholder,
            target=target_placeholder,
            angle=angle_placeholder,
            r=r_placeholder,
            target_angle=target_angle_placeholder,
            target_distance=target_distance_placeholder,
            dropout_keep_prob=dropout_keep_prob,
            input_size=input_size_placeholder,
            device="/gpu:0")

        return model

    def load_checkpoint(self, scene_scope):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        EXPERIMENT_CHECKPOINT_DIR = \
            "/home/aidas/Masters/target-driven-visual-navigation-metric-map/code/decoder/checkpoints/" +\
            DECODER_DIR + '/' + scene_scope

        checkpoint = tf.train.get_checkpoint_state(EXPERIMENT_CHECKPOINT_DIR)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
        else:
            print("Could not find old checkpoint")
        return sess

    def __global_step(self, epoch):
        mini_batch_count = self.training_data_size // MINI_BATCH_SIZE
        return (epoch + 1) * mini_batch_count * MINI_BATCH_SIZE
