"""Grid2DModelRendered allows to render an instance of Grid2DModel by using pygame."""

import pygame
import numpy as np


class Grid2DModelRenderer:
    def __init__(self, grid_2d_model):
        """

        :param grid_2d_model: Grid2DModel that needs to be rendered.

        Rendered grid is being scaled to appear bigger for convenience.
        Attribute scaling represents how many times a rendered grid should be bigger compared to a true grid, used by the agent.

        Entities in the grid are represented by RGB colors:

            Empty cells -- (255, 255, 255) white.
            Walls -- (0, 0, 0) black.
            Agent -- (255, 0, 0) red.
            Target -- (0, 255, 0) green.
            Cell in the observation -- (0, 0, 255) blue.
            Landmark -- random RGB color.
        """

        self.grid_model = grid_2d_model
        self.grid = self.grid_model.grid
        self.width = self.grid_model.shape[0]
        self.height = self.grid_model.shape[1]

        self.scaling = 50

        self.made_screen = False

        # Colors used in the grid.
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255,255,0)

    def render(self):
        """Renders the Grid2DModel using different colors for different entities."""
        if not self.made_screen:
            self.__init_screen()

        self.screen.fill((0, 0, 0))
        self.__render_grid_cells(self.__get_observed_cells())
        pygame.display.update()

    def close(self):
        if self.made_screen:
            pygame.quit()
            self.made_screen = False
        print("Closing Grid2DEnv")

    def __render_grid_cells(self, observations):
        for x in range(self.width):
            for y in range(self.height):
                self.__render_grid_cell(x, y, observations)

    def __render_grid_cell(self, x, y, observations):
        color = self.__get_grid_cell_color(x, y, observations)
        """For convenience self.height - 1 - y is used, since we want (0,0) coordinate to appear at the bottom left
        corner of the screen (which is consistent with usual xy-coordinate representation) instead of the top left
        corner."""
        pygame.draw.rect(
            self.screen, color,
            (x * self.scaling,
             (self.height - 1 - y) * self.scaling, self.scaling, self.scaling))

    def __get_grid_cell_color(self, x, y, observations):
        color = self.black
        if self.grid.get_location_value(x, y) == 0:
            color = self.white
        if self.grid_model.agent.get_pos() == (x, y):
            color = self.red
        if (x, y) in observations:
            color = self.blue
        if self.grid_model.target.get_pos() == (x, y):
            color = self.green
        if (x,y) in self.grid_model.grid.landmarks:
            color = tuple(np.random.choice(range(256), size=3))
        if len(self.grid_model.grid.landmarks) > 0 and (x,y) == self.grid_model.grid.landmarks[0]:
            color = self.yellow
        return color


    def __init_screen(self):
        pygame.init()
        screen_size = ((self.grid_model.shape[0]) * self.scaling,
                       (self.grid_model.shape[1]) * self.scaling)
        screen = pygame.display.set_mode(screen_size)
        self.screen = screen
        self.made_screen = True

    def __get_observed_cells(self):
        return self.grid_model.get_obs_cells()
