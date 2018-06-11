from .Grid import Grid
from utils.maze_generator import generate_random_maze_cells
import copy

class MazeGrid(Grid):
    def __init__(
            self,
            config,
            shape=(5, 5),
            cells=None):
        """ Initialize maze grid.

        :param shape: shape of the grid (number of rows, number of columns).
        :param cells: cells of the grid represented by matrix of entries of type Cell. Default is None.
            In case cells=None, grid is initialized as a randomly generated maze.

        If argument cells=None, then cells of the grid are initialized randomly using Recursive Backtracking Algorithm.
        """
        super().__init__(config,shape,cells)
        if not cells:
            self.cells = self.generate_random_grid()

        self.initial_cells = copy.deepcopy(self.cells)

    def generate_random_grid(self):
        return generate_random_maze_cells(self.cols, self.rows)