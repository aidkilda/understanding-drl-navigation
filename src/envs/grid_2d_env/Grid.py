from .Cell import Cell
from utils.maze_generator import generate_random_maze_cells
from .Target import Target
import numpy as np
import random
import copy


class Grid:
    """Represents a grid of the given shape.

    Grid is a matrix of numbers with default values:
        0 - means empty cell.
        1 - means wall.
        2 - means target.

    Out of bounds cells considered to be equivalent to wall cells.
    """

    def __init__(
            self,
            config,
            shape=(5, 5),
            cells=None):
        """ Initialize grid.

        :param shape: shape of the grid (number of rows, number of columns).
        :param cells: cells of the grid represented by matrix of entries of type Cell. Default is None.
        """
        self.config = config

        self.rows = shape[0]
        self.cols = shape[1]
        self.cells = copy.deepcopy(cells)
        self.random_reset = config['random_reset_grid']
        self.initial_cells = copy.deepcopy(self.cells)

        self.landmarks = []

    def reset(self):
        """Resets the grid's cells"""
        if self.random_reset:
            self.cells = self.generate_random_grid()
        else:
            self.cells = copy.deepcopy(self.initial_cells)

    def generate_random_grid(self):
        raise NotImplementedError

    @classmethod
    def create(cls, config):
        """Use @classmethod polymorphism to be able to construct Grid Objects Generically.

        :param config: configuration dictionary for arguments used to set up the right grid.
        :return: constructed Grid Object.
        """
        shape = config['shape']
        cells = config['initial_grid_cells']
        return cls(config, shape=shape, cells=cells)

    def locations_for_target(self):
        """
        Sometimes target can be placed only in restricted set of locations.
        :return: locations where target can be placed.
        """
        return self.empty_locations()

    def empty_locations(self):
        """
        :return: empty locations in the grid.
        """
        empty_locations = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c].is_empty():
                    empty_locations.append((r, c))
        return empty_locations

    def set_target(self, target_pos, can_step, is_transparent, value=2):
        """ Set target cell in the grid. Target cell is represented by number 2.

        :param target_pos: position of the target represented by tuple (x,y).
        :param can_step: whether agent can step on the target.
        :param is_transparent: whether agent can see through the target.
        :param value: numeric value for the target, as it should appear in agent's observation.
        """
        row = target_pos[0]
        col = target_pos[1]
        self.cells[row][col].value = value
        self.cells[row][col].can_step = can_step
        self.cells[row][col].is_transparent = is_transparent

    def get_location_value(self, row, col):
        """
        :return: value of a particular cell at (row, col) in the grid. Return 1 (means wall) for out of bounds cells.
        """
        if self.is_out_of_bounds_wall_location(row, col):
            return 1
        return self.cells[row][col].value

    def can_step_location(self, row, col):
        """
        :return: whether agent can step on a particular cell at (row, col) in the grid.
            Return False for out of bounds cells.
        """
        if self.is_out_of_bounds_wall_location(row, col):
            return False
        return self.cells[row][col].can_step

    def is_transparent_location(self, row, col):
        """
        :return: whether a particular cell at (row, col) in the grid is transparent.
            Return False for out of bounds cells.
        """
        if self.is_out_of_bounds_wall_location(row, col):
            return False
        return self.cells[row][col].is_transparent

    def get_random_empty_location(self):
        """
        :return: a random location in the grid, that is empty.
        """
        return random.choice(self.empty_locations())

    def is_out_of_bounds_wall_location(self, row, col):
        """
        :return: whether a given location (row,col) is out of bounds (hence considered to be a wall).
        """
        return row < 0 or row >= self.rows or col < 0 or col >= self.cols

    def get_random_location_for_target(self):
        """
        Sometimes target can be placed only in restricted set of locations.
        :return: a random location in the grid, where target could be placed.
        """
        return random.choice(self.locations_for_target())