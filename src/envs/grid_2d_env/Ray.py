import math


class Ray:
    """Represents a ray in a 2D plane.

    Ray is represented by equation: y = slope * x + intercept and direction given by angle theta.

    Edge cases arise when ray is horizontal (theta = 0 or theta = 180) or vertical (theta = 90 or theta = 270).
    Usually those conditions have to be checked and dealt with separately.
    """

    def __init__(self, start_point, theta):
        """Initializes ray.

        :param start_point: it could be any point on the ray. Used to derive the equation of the line.
        :param theta: angle (in degrees) of the ray with respect to x-axis. Angle increases by moving anti-clockwise.
        """
        self.start_point = start_point
        self.theta = theta

        self.slope = math.tan(math.radians(theta))
        self.intercept = start_point[1] - self.slope * start_point[0]

        self.is_vertical = False
        if theta == 90 or theta == 270:
            self.is_vertical = True

        self.is_horizontal = False
        if theta == 0 or theta == 180:
            self.is_horizontal = True

    def get_x(self, y):
        """
        :return: x coordinate of the point p on the ray, where p has y-coordinate y specified by the argument.
        """
        if self.is_vertical:
            return self.start_point[0]
        if self.is_horizontal:
            raise Exception('Horizontal line')
        return (y - self.intercept) / self.slope

    def get_y(self, x):
        """
        :return: y coordinate of the point p on the ray, where p has x-coordinate x specified by the argument.
        """
        if self.is_vertical:
            raise Exception('Vertical line')
        return self.slope * x + self.intercept

    def x_step(self):
        """
        :return: how x-coordinate changes by moving further along the ray.
        """
        if self.is_vertical:
            return 0
        elif 90 <= self.theta and self.theta < 270:
            return -1
        else:
            return 1

    def y_step(self):
        """
        :return: how y-coordinate changes by moving further along the ray.
        """
        if self.is_horizontal:
            return 0
        elif 0 <= self.theta and self.theta < 180:
            return 1
        else:
            return -1
