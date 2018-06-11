# -*- coding: utf-8 -*-
import numpy as np
import random
import math
from time import time
import os
import tensorflow as tf
import sys
import copy

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from internal_representation_analysis.decoder.internal_representation_prediction_analysis import visualize_coordinate_predictions
from internal_representation_analysis.decoder.internal_representation_prediction_analysis import visualize_angle_to_target_predictions
from internal_representation_analysis.decoder.internal_representation_prediction_analysis import get_error_info

from internal_representation_analysis.constants import SEED
from internal_representation_analysis.constants import TASK_LIST
from internal_representation_analysis.constants import DECODER_DIR
from position_decoder import PositionDecoderTrainer
from preprocess_decoder_data import DecoderDataPreprocessor

from internal_representation_analysis.scene_loader import THORDiscreteEnvironment as Environment


class PositionDecoderDataAnalyser(object):
    def __init__(self, scene_scope, trainer):
        self.scene_scope = scene_scope
        self.trainer = trainer

        self.model = None
        self.sess = None
        self.save_path = None

    def visualize_coordinate_results(self, dataset, save_dir):

        dataset = dataset[:50]
        predictions = self.get_prediction(self.list_state_embedding(dataset))

        pred_x = predictions[0]
        pred_y = predictions[1]

        real_x = self.list_state_x(dataset)
        real_y = self.list_state_y(dataset)

        visualize_coordinate_predictions(pred_x, pred_y, real_x, real_y,
                                     self.save_path + save_dir)

    def visualize_angle_to_target(self, dataset, save_dir):

        task = int(TASK_LIST[self.scene_scope][0])
        dataset = self.filter_by_target(dataset, task)
        dataset = dataset[:50]

        env = Environment({
            'scene_name': self.scene_scope,
            'terminal_state_id': dataset[0].target
        })

        target_x = env.get_x(dataset[0].target)
        target_y = env.get_y(dataset[0].target)
        predictions = self.get_prediction(self.list_state_embedding(dataset))
        real_angle = self.list_state_target_angle(dataset)
        real_x = self.list_state_x(dataset)
        real_y = self.list_state_y(dataset)

        visualize_angle_to_target_predictions(target_x, target_y, predictions,
                                  real_angle, real_x, real_y,
                                  self.save_path + save_dir)

    def get_prediction_info(self, dataset):
        accuracy = self.get_accuracy(dataset)
        for i in range(0, len(accuracy)//2):
            print("Mean of %d: %f" % (i, accuracy[2 * i]))
            print("Std of %d: %f" % (i, np.sqrt(accuracy[(2*i)+1])))

    def get_accuracy(self, dataset):
        feed_dict = self.trainer.get_feed_dict(self.model, dataset)
        return self.sess.run(self.model.accuracy, feed_dict=feed_dict)

    def get_prediction(self, embeddings):
        pred = self.sess.run(
            self.model.prediction,
            feed_dict={
                self.model.embedding: embeddings
            })
        return pred

    def get_midpoint_info(self, dataset):
        real_x = self.__get_midpoints(self.list_state_x(dataset))
        real_y = self.__get_midpoints(self.list_state_y(dataset))

        mid_pred = self.get_prediction(
            self.__get_midpoints(self.list_state_embedding(dataset)))
        pred_x = mid_pred[0]
        pred_y = mid_pred[1]

        get_error_info(pred_x, pred_y, real_x, real_y)

    def __get_midpoints(self, points):
        midp = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                midp.append(self.__get_midpoint(points[i], points[j]).tolist())
        return midp

    def __get_midpoint(self, p1, p2):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        return np.true_divide(p1 + p2, 2)

    def list_state_x(self, dataset):
        return [state.x for state in dataset]

    def list_state_y(self, dataset):
        return [state.y for state in dataset]

    def list_state_theta(self, dataset):
        return [state.theta for state in dataset]

    def list_state_embedding(self, dataset):
        return [state.embedding for state in dataset]

    def list_state_target(self, dataset):
        return [state.target for state in dataset]

    def list_state_target_angle(self, dataset):
        return [state.target_angle for state in dataset]

    def list_state_target_distance(self, dataset):
        return [state.target_distance for state in dataset]

    def filter_by_target(self, dataset, target):
        return [dataset[i] for i, v in enumerate(dataset) if v.target==target]

    def filter_by_theta(self, dataset, theta):
        return [dataset[i] for i, v in enumerate(dataset) if v.theta==theta]

    def analyze_results(self):

        self.save_path = 'decoding_results/' + DECODER_DIR + '/' + self.scene_scope + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        sys.stdout = open(self.save_path + 'numbers.txt', 'w')

        self.model = self.trainer.create_model()
        self.sess = self.trainer.load_checkpoint(self.scene_scope)

        print("Validation:")
        self.get_prediction_info(self.trainer.state_dataset.validation_set)
        #self.visualize_angle_to_target(copy.deepcopy(self.trainer.state_dataset.validation_set), 'plots/val/')
        self.visualize_coordinate_results(copy.deepcopy(self.trainer.state_dataset.validation_set), 'plots/val/')
        self.get_midpoint_info(self.trainer.state_dataset.validation_set)

        print("Training:")
        self.get_prediction_info(self.trainer.state_dataset.train_set)
        #self.visualize_angle_to_target(copy.deepcopy(self.trainer.state_dataset.train_set), 'plots/train/')
        self.visualize_coordinate_results(copy.deepcopy(self.trainer.state_dataset.train_set), 'plots/train/')
        self.get_midpoint_info(self.trainer.state_dataset.train_set)

        print("Testing:")
        self.get_prediction_info(self.trainer.state_dataset.test_set)
        #self.visualize_angle_to_target(copy.deepcopy(self.trainer.state_dataset.test_set), 'plots/test/')
        self.visualize_coordinate_results(copy.deepcopy(self.trainer.state_dataset.test_set), 'plots/test/')
        self.get_midpoint_info(self.trainer.state_dataset.test_set)

    ''' VISUALIZATIONS'''

    def plot_pca(self, scene_scope, tasks):
        tasks.append(None)
        for task in tasks:
            task_name = task
            if not task:
                task_name = 'all_targets'
            save_path = 'pca_plots/' + 'scene_spec_/' + scene_scope + '/task_' + task_name + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            dataset = self.trainer.state_dataset.all_states
            if task:
                dataset = self.filter_by_target(self.trainer.state_dataset.all_states, int(task))

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(
                self.list_state_embedding(dataset))
            print('Explained variation per principal component: {}'.format\
            (pca.explained_variance_ratio_))

            df = self.make_data_frame(dataset=dataset)
            df['pca-one'] = pca_result[:, 0]
            df['pca-two'] = pca_result[:, 1]

            # define the colormap
            cmap = plt.cm.jet

            bounds_x = [0] + [i + 0.25 for i in sorted(list(set(df['x'])))]
            norm_x = colors.BoundaryNorm(bounds_x, cmap.N)

            bounds_y = [0] + [i + 0.25 for i in sorted(list(set(df['y'])))]
            norm_y = colors.BoundaryNorm(bounds_y, cmap.N)

            bounds_theta = [0] + [i + 1 for i in sorted(list(set(df['theta'])))]
            norm_theta = colors.BoundaryNorm(bounds_theta, cmap.N)

            bounds_target = [0] + [i + 1 for i in sorted(list(set(df['target'])))]
            norm_target = colors.BoundaryNorm(bounds_target, cmap.N)

            chart_x = df.plot.scatter(
                x='pca-one', y='pca-two', c='x', cmap=cmap, norm=norm_x, s=5)
            plt.savefig(save_path + 'x.png')

            chart_y = df.plot.scatter(
                x='pca-one', y='pca-two', c='y', cmap=cmap, norm=norm_y, s=5)
            plt.savefig(save_path + 'y.png')

            chart_theta = df.plot.scatter(
                x='pca-one',
                y='pca-two',
                c='theta',
                cmap=cmap,
                norm=norm_theta,
                s=5)
            plt.savefig(save_path + 'theta.png')

            chart_target_angle = df.plot.scatter(
                x='pca-one', y='pca-two', c='target_angle', colormap='viridis', s=5)
            plt.savefig(save_path + 'target_angle.png')

            chart_target_distance = df.plot.scatter(
                x='pca-one', y='pca-two', c='target_distance', colormap='viridis', s=5)
            plt.savefig(save_path + 'target_distance.png')

            if not task:
                chart_target = df.plot.scatter(
                    x='pca-one', y='pca-two', c='target', cmap=cmap, norm=norm_target, s=5)
                plt.savefig(save_path + 'target.png')


    def plot_tsne(self, scene_scope, tasks):
        perplexities = [5,30,50,100]
        thetas = [0,90,180,270]
        tasks.append(None)

        for task in tasks:
            for theta in thetas:
                for p in perplexities:
                    task_name = task
                    if not task:
                        task_name = 'all_targets'
                    save_path = 'tsne_plots/' +'scene_spec_theta/'+ scene_scope + '/task_' + task_name + \
                                '/theta_' + str(theta) +'/perplexity_' + str(p) + '/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    dataset = self.trainer.state_dataset.all_states
                    if task:
                        dataset = self.filter_by_target(self.trainer.state_dataset.all_states, int(task))

                    df = self.make_data_frame(dataset=dataset)
                    tsne = TSNE(perplexity=p)
                    tsne_results = tsne.fit_transform(
                        self.list_state_embedding(dataset))
                    df_tsne = df.copy()
                    df_tsne['x-tsne'] = tsne_results[:, 0]
                    df_tsne['y-tsne'] = tsne_results[:, 1]

                    # define the colormap
                    cmap = plt.cm.jet

                    bounds_x = [0] + [i + 0.25 for i in sorted(list(set(df['x'])))]
                    norm_x = colors.BoundaryNorm(bounds_x, cmap.N)

                    bounds_y = [0] + [i + 0.25 for i in sorted(list(set(df['y'])))]
                    norm_y = colors.BoundaryNorm(bounds_y, cmap.N)

                    bounds_theta = [0] + [i + 1 for i in sorted(list(set(df['theta'])))]
                    norm_theta = colors.BoundaryNorm(bounds_theta, cmap.N)

                    bounds_target = [0] + [i + 1 for i in sorted(list(set(df['target'])))]
                    norm_target = colors.BoundaryNorm(bounds_target, cmap.N)


                    chart_x = df_tsne.plot.scatter(
                        x='x-tsne', y='y-tsne', c='x', cmap=cmap, norm=norm_x, s=5)
                    plt.savefig(save_path + 'x.png')

                    chart_y = df_tsne.plot.scatter(
                        x='x-tsne', y='y-tsne', c='y', cmap=cmap, norm=norm_y, s=5)
                    plt.savefig(save_path + 'y.png')

                    chart_theta = df_tsne.plot.scatter(
                        x='x-tsne', y='y-tsne', c='theta', cmap=cmap, norm=norm_theta, s=5)
                    plt.savefig(save_path + 'theta.png')

                    chart_target_angle = df_tsne.plot.scatter(
                        x='x-tsne', y='y-tsne', c='target_angle', colormap='magma', s=5)
                    plt.savefig(save_path + 'target_angle.png')

                    chart_target_distance = df_tsne.plot.scatter(
                        x='x-tsne', y='y-tsne', c='target_distance', colormap='magma', s=5)
                    plt.savefig(save_path + 'target_distance.png')

                    if not task:
                        chart_target = df_tsne.plot.scatter(
                            x='x-tsne', y='y-tsne', c='target', cmap=cmap, norm=norm_target, s=5)
                        plt.savefig(save_path + 'target.png')
                    print("Done " + save_path)

    def make_data_frame(self, dataset):
        d = {
            'emb':
            self.list_state_embedding(dataset),
            'x':
            self.list_state_x(dataset),
            'y':
            self.list_state_y(dataset),
            'theta':
            self.list_state_theta(dataset),
            'target':
            self.list_state_target(dataset),
            'target_angle':
            self.list_state_target_angle(dataset),
            'target_distance':
            self.list_state_target_distance(dataset)
        }
        return pd.DataFrame(d)

def main():
    np.random.seed(SEED)
    random.seed(SEED)
    tf.set_random_seed(SEED)
    start_time = time()
    data_preprocessor = DecoderDataPreprocessor()
    state_datasets = data_preprocessor.get_data_for_decoder()
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()
    for scene_scope in scene_scopes:
        state_datasets[scene_scope].split_datasets(seed=SEED, all_targets=True, test_target_eq_obs=False)
        print("Scene scope: ", scene_scope)
        print("Sample size: ", len(state_datasets[scene_scope].all_states))
        trainer = PositionDecoderTrainer(
            scene_scope=scene_scope, state_dataset=state_datasets[scene_scope])
        analyser = PositionDecoderDataAnalyser(
            scene_scope=scene_scope, trainer=trainer)

        #analyser.trainer.train(True)
        tf.reset_default_graph()
        analyser.analyze_results()

        # tasks = list_of_tasks[scene_scope]
        # print("t-SNE")
        # analyser.plot_tsne(scene_scope, tasks)
        #print("PCA")
        #analyser.plot_pca(scene_scope, tasks)


if __name__ == '__main__':
    main()
