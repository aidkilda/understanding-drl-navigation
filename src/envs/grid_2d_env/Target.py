"""Target encapsulates the concept of a target in the environment."""

class Target:

    def __init__(self, config, grid):
        """Initializes target.

        :param config: configuration dictionary for arguments used to set up the right target.
        :param grid: grid where target is located.
        """
        position = config['initial_target']

        if not config['initial_target']:
            position = grid.get_random_location_for_target()

        self.x, self.y = position

        self.random_reset = config['random_reset_target']

    def reset(self, grid):
        """Resets the target's position. Note that it depends on the current grid."""
        if self.random_reset:
            self.x, self.y = grid.get_random_location_for_target()

    def get_pos(self):
        return self.x, self.y