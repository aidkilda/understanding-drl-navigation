import numpy as np
import matplotlib.pyplot as plt


class SceneVisualizer(object):
    def __init__(self, env):

        self.env = env
        self.scene_scope = env.scene_name
        self.locations_x = [l[0] for l in env.locations]
        self.locations_y = [l[1] for l in env.locations]
        self.target_x = env.get_x(int(env.terminal_state_id))
        self.target_y = env.get_y(int(env.terminal_state_id))
        self.target_theta = env.get_rotation(int(env.terminal_state_id))

    def visualize_trajectory(self, list_env_id):
        plt.title(self.scene_scope + " " + str(self.target_x) + ", " +
                  str(self.target_y))
        self.__draw_map()
        #print("n loc in", self.env.n_locations)
        #print("list env id", list_env_id)
        for _, id in enumerate(list_env_id):
            print(id)
            self.__mark_pos(
                self.env.get_x(id), self.env.get_y(id),
                self.env.get_rotation(id))
        # Mark starting position
        self.__mark_pos(
            self.env.get_x(list_env_id[0]),
            self.env.get_y(list_env_id[0]),
            self.env.get_rotation(list_env_id[0]),
            color="green")
        # Mark target
        self.__mark_pos(
            self.target_x, self.target_y, self.target_theta, color="red")
        plt.show()

    def visualize_scene(self):
        self.__draw_map()
        self.__mark_target()
        self.__mark_pos(self.env.x, self.env.y, self.env.r)
        plt.show()

    def save_scene(self, x, y, r, save_path):
        self.__draw_map()
        self.__mark_target()
        self.__mark_pos(x, y, r)
        plt.savefig(save_path)
        plt.close()

    def mark_targets(self, targets, color="orange"):
        for target in targets:
            self.__mark_pos(self.env.get_x(int(target)),
                            self.env.get_y(int(target)),
                            self.env.get_rotation(int(target)),
                            color=color)

    def __draw_map(self):
        plt.scatter(self.locations_x, self.locations_y, color="black")

    def __mark_target(self):
        plt.scatter(self.target_x, self.target_y, color="red")
        self.__draw_curr_angle(
            self.target_x, self.target_y, self.target_theta, color="red")

    def __mark_pos(self, x, y, theta, color="blue"):
        plt.scatter(x, y, color=color)
        self.__draw_curr_angle(x, y, theta, color=color)

    #TODO(aidkilda) make code accomodate angles different than multiples of 90.
    def __draw_curr_angle(self,
                          curr_pos_x,
                          curr_pos_y,
                          curr_angle,
                          color="black"):
        #print("Curr angle", curr_angle)
        if self.__isclose(curr_angle, 0.0, rel_tol=1e-5):
            dx = 0.0
            dy = 0.2
        elif self.__isclose(curr_angle, 90.0, rel_tol=1e-5):
            dx = 0.2
            dy = 0.0
        elif self.__isclose(curr_angle, 180.0, rel_tol=1e-5):
            dx = 0.0
            dy = -0.2
        elif self.__isclose(curr_angle, 270.0, rel_tol=1e-5):
            dx = -0.2
            dy = 0.0
        else:  # Don't draw arrow at all
            return
        plt.arrow(
            curr_pos_x,
            curr_pos_y,
            dx,
            dy,
            head_width=0.05,
            head_length=0.1,
            color=color)

    def __draw_curr_angle_flex(self,
                          curr_pos_x,
                          curr_pos_y,
                          curr_angle,
                          color="black"):
        #print("Curr angle", curr_angle)
        if self.__isclose(curr_angle, 0.0, rel_tol=1e-5):
            dx = 0.0
            dy = 0.2
        elif self.__isclose(curr_angle, 90.0, rel_tol=1e-5):
            dx = 0.2
            dy = 0.0
        elif self.__isclose(curr_angle, 180.0, rel_tol=1e-5):
            dx = 0.0
            dy = -0.2
        elif self.__isclose(curr_angle, 270.0, rel_tol=1e-5):
            dx = -0.2
            dy = 0.0
        else:  # Don't draw arrow at all
            return
        plt.arrow(
            curr_pos_x,
            curr_pos_y,
            dx,
            dy,
            head_width=0.05,
            head_length=0.1,
            color=color)

    # Used for equality comparisson of floats. Taken from documentation.
    def __isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
