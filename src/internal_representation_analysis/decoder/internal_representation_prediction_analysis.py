"""
The file contains functions that can be used for analyzing navigating agent's internal representation of the
environment.
"""
import numpy as np
import random
import math
import os
import tensorflow as tf
import sys
import copy
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *
from matplotlib import colors
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from internal_representation_analysis.utils.tools import SimpleImageViewer
from internal_representation_analysis.decoder.scene_visualizer import SceneVisualizer

# Visualizing prediction errors.

def visualize_coordinate_results(pred_x, pred_y, real_x, real_y, save_dir):
    for i in range(len(pred_x)):
        x = [real_x[i], pred_x[i]]
        y = [real_y[i], pred_y[i]]
        plt.scatter(real_x[i], real_y[i], color="green")
        plt.scatter(pred_x[i], pred_y[i], color="red")
        plt.plot(x, y, color="black")

    __save_figure(save_dir, 'xy.png')
    __save_figure(save_dir, 'xy.pdf')
    plt.close()


def visualize_angle_to_target(target_x, target_y, pred_angle, real_angle, real_x, real_y, save_dir):
    for i in range(len(pred_angle)):
        plt.scatter(real_x[i], real_y[i], color="black")
        __plot_point(real_x[i], real_y[i], real_angle[i], color="green")
        __plot_point(real_x[i], real_y[i], pred_angle[i], color="red")

    plt.scatter(target_x, target_y, color="red")

    __save_figure(save_dir, 'angle_to_target.png')
    __save_figure(save_dir, 'angle_to_target.pdf')
    plt.close()

def __plot_point(x, y, angle, color):
    endy = y + 0.5 * math.sin(math.radians(angle))
    endx = x + 0.5 * math.cos(math.radians(angle))

    plt.plot([x, endx], [y, endy], color=color)

def __save_figure(save_dir, save_file):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + save_file)


# Midpoint prediction.

def get_error_info(pred_x, pred_y, real_x, real_y):
    errors_x = [abs(real_x[i] - pred_x[i]) for i in range(len(pred_x))]
    errors_y = [abs(real_y[i] - pred_y[i]) for i in range(len(pred_y))]
    __print_mean_std(errors_x)
    __print_mean_std(errors_y)

def __print_mean_std(nums):
    print("Mean err mid y", np.mean(nums))
    print("Std err mid y", np.std(nums))


