# -*- coding: utf-8 -*-

from internal_representation_analysis.decoder.utils.decoder_network_utils import define_scope


# Regressor network for postion decoding
class DecoderModel(object):
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
        self._device = device
        self.scene_scope = scene_scope
        self.x = x
        self.y = y
        self.theta = theta
        self.label = label
        self.embedding = embedding
        self.target = target
        self.angle = angle
        self.r = r
        self.target_angle = target_angle
        self.target_distance = target_distance
        self.dropout_keep_prob = dropout_keep_prob
        self.input_size=input_size
        self.prediction
        self.loss
        self.optimize
        self.summary

    @define_scope
    def prediction(self):
        raise NotImplementedError()

    @define_scope
    def loss(self):
        raise NotImplementedError()

    @define_scope
    def optimize(self):
        raise NotImplementedError()

    @define_scope
    def summary(self):
        raise NotImplementedError()

    @define_scope
    def accuracy(self):
        raise NotImplementedError()
