from .Ray import Ray
import math


class VisibleCellsFinder:
    """VisibleCellsFinder enables to search how many visible cells (cells that are not blocked by any obstacles)
     are in particular direction(s)"""

    def get_visible_cells_along_rays(self, grid, angles,
                                   agent_position_and_rotation):
        """Get all visible cells that lie at one of the given angles from the agent's current position.

        :param grid: grid where agent navigates.
        :param angles: list of all angles, at which we draw rays from the agent's current position and rotation.
        :param agent_position_and_rotation: current position and rotation of the agent.
        :return: a list of all visible cells that lie at one of the given angles from the agent's current position.
        """
        a_x, a_y = agent_position_and_rotation.get_pos()
        a_rotation = agent_position_and_rotation.rotation
        visible_cells_along_rays = []

        for angle in angles:
            # grid-world lines are drawn from the midpoint of a cell (i.e. for a cell (x,y),
            # a corresponding start point of a line would be (x + 1/2, y + 1/2)).
            ray = Ray((a_x + 1 / 2, a_y + 1 / 2), (a_rotation + angle) % 360)
            visible_cells_for_angle = self.get_visible_cells_along_ray(
                grid, ray, (a_x, a_y))
            visible_cells_along_rays.append(visible_cells_for_angle)

        return visible_cells_along_rays

    def get_visible_cells_along_ray(self, grid, ray, start_cell):
        """

        :param grid: grid where agent navigates.
        :param ray: represented by object of type Ray.
        :param start_cell: cell from which the ray is drawn.
        :return: a list of coordinates of all visible cells that lie along the given ray starting at start_cell.
        """
        visible_cells = []
        x, y = self.__get_next_cell_on_ray(ray, start_cell[0], start_cell[1])
        while grid.is_transparent_location(x, y):
            visible_cells.append((x, y))
            x, y = self.__get_next_cell_on_ray(ray, x, y)
        # Add the last visible cell (obstacle). Note that it might be a out of bounds wall cell.
        visible_cells.append((x, y))
        return visible_cells

    def get_num_visible_cells_along_rays(self, grid, angles,
                                       agent_position_and_rotation):
        """Get numbers of visible cells that lie at one of the given angles from the agent's current position.

        :param grid: grid where agent navigates.
        :param angles: list of all angles, at which we draw rays from the agent's current position and rotation.
        :param agent_position_and_rotation: current position and rotation of the agent.
        :return: list of numbers of visible cells that lie along each ray at given angles.
        """
        a_x, a_y = agent_position_and_rotation.get_pos()
        a_rotation = agent_position_and_rotation.rotation
        visible_cells = []

        for angle in angles:
            # grid-world lines are drawn from the midpoint of a cell (i.e. for a cell (x,y),
            # a corresponding start point of a line would be (x + 1/2, y + 1/2)).
            ray = Ray((a_x + 1 / 2, a_y + 1 / 2), (a_rotation + angle) % 360)
            num_visible_cells_for_angle = self.get_num_visible_cells_along_ray(
                grid, ray, (a_x, a_y))
            visible_cells.append(num_visible_cells_for_angle)

        return visible_cells

    def get_num_visible_cells_along_ray(self, grid, ray, start_cell):
        """

        :param grid: grid where agent navigates.
        :param ray: represented by object of type Ray.
        :param start_cell: cell from which the ray is drawn.
        :return: a number of all visible cells that lie along the given ray starting at start_cell.
        """
        return len(self.get_visible_cells_along_ray(grid, ray, start_cell))

    def __get_next_cell_on_ray(self, ray, x, y):
        x_step = ray.x_step()
        y_step = ray.y_step()

        x_next = x + x_step
        y_next = y + y_step

        # If x_step or y_step <= 0, then next coordinate we consider should be current coordinate.
        # This happens, because we draw rays from points (x + 1/2, y + 1/2) in the grid.

        if x_step <= 0:
            x_next = x

        if y_step <= 0:
            y_next = y

        # Check which coordinate changes first (or they both change at the same time, which corresponds to move along
        # the diagonal in the grid).
        if not ray.is_vertical and math.floor(ray.get_y(x_next)) == y:
            x += x_step
        elif not ray.is_horizontal and math.floor(ray.get_x(y_next)) == x:
            y += y_step
        else:
            x += x_step
            y += y_step

        return x, y
