"""Abstraction for a grid cell."""
class Cell:

    def __init__(self,
                 value,
                 can_step,
                 is_transparent=True):
        """Initializes cell.

        :param value: value of the cell. Purpose is that different kinds of cells, such as target, landmarks, walls,
            etc., could have different values representing them.
        :param can_step: boolean indicating whether agent can step on this cell.
        :param is_transparent: boolean indicating whether agent can see through the cell.
        """
        self.value = value
        self.can_step = can_step
        self.is_transparent = is_transparent

    def is_empty(self):
        """Cell is considered to be empty if its value is equal to 0."""
        return self.value == 0