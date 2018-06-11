class Observation:
    """Observation is an abstract class for representing an observation in the environment."""

    def get(self, agent_position_and_rotation, grid):
        """ Returns observation """
        raise NotImplementedError

    def get_cells(self, agent_position_and_rotation, grid):
        """ Returns grid cells, that are observed by the agent. """
        raise NotImplementedError

    def size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    @classmethod
    def create(cls, config):
        """Use @classmethod polymorphism to be able to construct Observation Objects Generically.

        :param config: configuration dictionary for arguments used to set up the right observation.
        :return: constructed Observation Object.
        """
        raise NotImplementedError
