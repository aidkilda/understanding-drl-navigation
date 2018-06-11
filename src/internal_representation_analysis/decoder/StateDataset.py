import random

from internal_representation_analysis.network import ActorCriticFFNetwork
from internal_representation_analysis.scene_loader import THORDiscreteEnvironment as Environment

from internal_representation_analysis.constants import MINI_BATCH_SIZE


class StateDataset(object):
    def __init__(self, states):

        self.all_states = states
        self.train_set = None
        self.validation_set = None
        self.test_set = None

    def __eq__(self, other):
        return self.all_states == other.all_states

    def split_datasets(self, seed, all_targets=False, test_target_eq_obs=False):
        all_states = self.all_states[:]
        random.seed(seed)
        random.shuffle(all_states)

        if test_target_eq_obs:
            for s in all_states:
                s.embedding = s.target_eq_obs

        if not all_targets:
            self.train_set = all_states[0:int(0.6 * len(all_states))]
            self.validation_set = all_states[int(0.6 * len(all_states)):int(
                0.8 * len(all_states))]
            self.test_set = all_states[int(0.8 * len(all_states)):]
        else:
            unique_state_ids = list(set([s.state_id for s in all_states]))
            random.shuffle(unique_state_ids)
            train_ids = set(unique_state_ids[0:int(0.6 * len(unique_state_ids))])
            val_ids = set(unique_state_ids[int(0.6 * len(unique_state_ids)):int(
                0.8 * len(unique_state_ids))])
            test_ids = set(unique_state_ids[int(0.8 * len(unique_state_ids)):])

            self.train_set = [s for s in all_states if s.state_id in train_ids]
            self.validation_set = [s for s in all_states if s.state_id in val_ids]
            self.test_set = [s for s in all_states if s.state_id in test_ids]

    def shuffle_train_set(self):
        random.shuffle(self.train_set)

    def get_train_mini_batch(self, start_index):
        return self.train_set[start_index:start_index + MINI_BATCH_SIZE]

    def filter_by_indexes(self, indexList):
        self.all_states = [self.all_states[i] for i in indexList]
