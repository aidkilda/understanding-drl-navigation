from .Grid import Grid
from envs.grid_2d_env.Cell import Cell
import copy

class SingleLandmarkGrid(Grid):
    def __init__(
            self,
            config,
            shape=(10, 10),
            cells=None):
        """ Initialize single landmark grid.

        :param shape: shape of the grid (number of rows, number of columns).
        :param cells: cells of the grid represented by matrix of entries of type Cell. Default is None.
        """
        self.num_landmarks = config['num_landmarks']
        self.target_distance_from_beacon = config['target_distance_from_beacon']
        self.SIDE_COLOR = 3
        super().__init__(config, shape, cells)
        if not cells:
            self.cells = self.generate_random_grid()

        self.initial_cells = copy.deepcopy(self.cells)

    #TODO(aidkilda) refactor this method.
    def generate_random_grid(self):
        self.landmarks = []
        self.cells = self.__generate_empty_grid()
        self.__add_colored_side()
        self.__randomly_place_landmarks()
        return self.cells

    def locations_for_target(self):
        """
        Target's location is fully determined by the landmark configuration.
        :return: location where target can be placed.
        """
        target_row = self.landmarks[0][0]
        target_col = self.landmarks[0][1] - self.target_distance_from_beacon
        return [(target_row, target_col)]

    def __generate_empty_grid(self):
        cells = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(Cell(0, True, True))
            cells.append(row)
        return cells

    def __add_colored_side(self):
        """Add a colored side to the grid, providing strong directional reference."""
        for r in range(self.rows):
            self.cells[r][self.cols - 1].value = self.SIDE_COLOR
            self.cells[r][self.cols - 1].can_step = False

    def __randomly_place_beacon(self):
        if self.config['beacon_loc']:
            row, col = self.config['beacon_loc']
        else:
            row, col = self.get_random_empty_location()

        while col - self.target_distance_from_beacon < 0:
            row, col = self.get_random_empty_location()

        self.landmarks.append((row,col))
        self.cells[row][col].value = self.SIDE_COLOR + 1
        self.cells[row][col].can_step = False
        self.cells[row][col].is_transparent = False

    def __randomly_place_landmarks(self):
        self.__randomly_place_beacon()
        for i in range(1, self.num_landmarks):
            row, col = self.get_random_empty_location()

            self.landmarks.append((row, col))
            self.cells[row][col].value = self.SIDE_COLOR + 1 + i
            self.cells[row][col].can_step = False
            self.cells[row][col].is_transparent = False
