# -*- coding: utf-8 -*-
import math
import numpy as np
import pickle

from State import State
from sklearn import preprocessing

from internal_representation_analysis.constants import LABELS_FILE
from internal_representation_analysis.constants import ROTATION_FILE
from internal_representation_analysis.constants import SPEC_LAYER_FILE
from internal_representation_analysis.constants import EMBEDDINGS_FILE
from internal_representation_analysis.constants import TARGET_FILE
from internal_representation_analysis.constants import TARGET_X_FILE
from internal_representation_analysis.constants import TARGET_Y_FILE
from internal_representation_analysis.constants import TASK_LIST
from internal_representation_analysis.constants import X_COORD_FILE
from internal_representation_analysis.constants import Y_COORD_FILE
from internal_representation_analysis.constants import STATE_ID_FILE
from internal_representation_analysis.constants import TARGET_EQ_OBS_FILE
from StateDataset import StateDataset


class DecoderDataPreprocessor:

    def __init__(self):
        self.data_files = [SPEC_LAYER_FILE, LABELS_FILE, X_COORD_FILE, Y_COORD_FILE, ROTATION_FILE,
                           TARGET_FILE, TARGET_X_FILE, TARGET_Y_FILE, STATE_ID_FILE, TARGET_EQ_OBS_FILE]

    def get_data_for_decoder(self):
        return self.__get_data(self.data_files)

    def __get_data(self, data_files):
        data_items_list = []
        for file in data_files:
            data_item_str = open(file, 'rb')
            data_items_list.append(pickle.load(data_item_str))

        scene_scopes = TASK_LIST.keys()
        return self.create_state_datasets(*data_items_list, scene_scopes=scene_scopes)

    def create_state_datasets(self, embeddings, labels, x, y, theta, target,
                              target_x, target_y, state_id, target_eq_obs, scene_scopes):
        state_datasets = {}
        for scene_scope in scene_scopes:
            state_dataset = self.__create_state_dataset(
                x[scene_scope], y[scene_scope], theta[scene_scope],
                labels[scene_scope], embeddings[scene_scope],
                target[scene_scope], target_x[scene_scope],
                target_y[scene_scope], state_id[scene_scope], target_eq_obs[scene_scope])
            state_datasets[scene_scope] = state_dataset
        return state_datasets

    def __create_state_dataset(self, x, y, theta, labels, embeddings, target,
                               target_x, target_y, state_id, target_eq_obs):
        states = [
            State(t[0], t[1], t[2], t[3], t[4], t[9], t[5], t[8],
                  self.angle_between((t[0], t[1]), (0, 0)),
                  self.distance_between(t[0], t[1], 0, 0),
                  self.angle_between((t[6]-t[0], t[7]-t[1]), (0, 0)),
                  self.distance_between(t[0], t[1], t[6], t[7]))
            for t in zip(x, y, theta, labels, embeddings, target, target_x,
                         target_y, state_id, target_eq_obs)
        ]
        return StateDataset(states)

    def distance_between(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def angle_between(self, p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    def standartize(self, dataset):
        preprocessing.scale(dataset)

