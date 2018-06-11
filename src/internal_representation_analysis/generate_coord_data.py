#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle

from internal_representation_analysis.network import ActorCriticFFNetwork
from internal_representation_analysis.scene_loader import THORDiscreteEnvironment as Environment

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR

from constants import TASK_TYPE
from constants import TASK_LIST
from constants import USE_GPU

from constants import EMBEDDINGS_FILE
from constants import LABELS_FILE
from constants import X_COORD_FILE
from constants import Y_COORD_FILE
from constants import ROTATION_FILE
from constants import SPEC_LAYER_FILE
from constants import TARGET_FILE
from constants import TARGET_X_FILE
from constants import TARGET_Y_FILE
from constants import STATE_ID_FILE
from constants import TARGET_EQ_OBS_FILE


def one_hot_encode(label, depth):
    one_hot = np.zeros(depth)
    one_hot[label] = 1
    return one_hot.tolist()


def write_to_file(filepath, data):
    target_file = open(filepath, 'wb')
    pickle.dump(data, target_file)
    target_file.close()
    print("Finished writing to a file %s" % (filepath))


# Environment's locations represent combination of (coordinates, rotation), where rotation has 4 possibilities
# We only care about coordinates, hence we treat all rotations at the same coordinates as the same location
def get_num_locations(env):
    return env.n_locations // 4


def get_location_id(current_state_id):
    return current_state_id // 4


if __name__ == '__main__':

    device = "/gpu:0" if USE_GPU else "/cpu:0"
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()

    global_network = ActorCriticFFNetwork(
        action_size=ACTION_SIZE,
        device=device,
        network_scope=network_scope,
        scene_scopes=scene_scopes)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")

    scene_embed = dict()
    scene_pos_label = dict()
    scene_pos_x = dict()
    scene_pos_y = dict()
    scene_pos_theta = dict()
    scene_spec_layer = dict()
    scene_target = dict()
    scene_target_x = dict()
    scene_target_y = dict()
    scene_state_id = dict()
    scene_target_eq_obs = dict()

    for scene_scope in scene_scopes:

        scene_embed[scene_scope] = []
        scene_pos_label[scene_scope] = []
        scene_pos_x[scene_scope] = []
        scene_pos_y[scene_scope] = []
        scene_pos_theta[scene_scope] = []
        scene_spec_layer[scene_scope] = []
        scene_target[scene_scope] = []
        scene_target_x[scene_scope] = []
        scene_target_y[scene_scope] = []
        scene_state_id[scene_scope] = []
        scene_target_eq_obs[scene_scope] = []

        for task_scope in list_of_tasks[scene_scope]:

            env = Environment({
                'scene_name': scene_scope,
                'terminal_state_id': int(task_scope)
            })
            print("Task", task_scope, "task location: ",
                  env.locations[int(task_scope)], env.rotations[int(task_scope)], int(task_scope))

            scopes = [network_scope, scene_scope]

            for current_state_id in range(env.n_locations):
                scene_pos_label[scene_scope].append(
                    one_hot_encode(
                        get_location_id(current_state_id),
                        get_num_locations(env)))
                scene_pos_x[scene_scope].append(env.get_x(current_state_id))
                scene_pos_y[scene_scope].append(env.get_y(current_state_id))
                scene_pos_theta[scene_scope].append(
                    env.get_rotation(current_state_id))
                state = env.get_state(current_state_id)
                embedding = global_network.run_observation_embedding(
                    sess, state, scopes)
                scene_embed[scene_scope].append(embedding)
                spec_layer = global_network.run_scene_specific_layer(
                    sess, state, env.target, scopes)
                scene_spec_layer[scene_scope].append(spec_layer)

                target_eq_obs = global_network.run_scene_specific_layer(
                    sess, state, state, scopes)
                scene_target_eq_obs[scene_scope].append(target_eq_obs)

                scene_target[scene_scope].append(int(task_scope))
                scene_target_x[scene_scope].append(env.get_x(int(task_scope)))
                scene_target_y[scene_scope].append(env.get_y(int(task_scope)))

                scene_state_id[scene_scope].append(current_state_id)
        print("Finished with scene %s. Total %d observations" %
              (scene_scope, len(scene_embed[scene_scope])))

write_to_file('decoder/' + EMBEDDINGS_FILE, scene_embed)
write_to_file('decoder/' + LABELS_FILE, scene_pos_label)
write_to_file('decoder/' + X_COORD_FILE, scene_pos_x)
write_to_file('decoder/' + Y_COORD_FILE, scene_pos_y)
write_to_file('decoder/' + ROTATION_FILE, scene_pos_theta)
write_to_file('decoder/' + SPEC_LAYER_FILE, scene_spec_layer)
write_to_file('decoder/' + TARGET_FILE, scene_target)
write_to_file('decoder/' + TARGET_X_FILE, scene_target_x)
write_to_file('decoder/' + TARGET_Y_FILE, scene_target_y)
write_to_file('decoder/' + STATE_ID_FILE, scene_state_id)
write_to_file('decoder/' + TARGET_EQ_OBS_FILE, scene_target_eq_obs)
