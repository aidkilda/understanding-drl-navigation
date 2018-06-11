from .PositionAndRotation import PositionAndRotation


class Action:
    """Action encapsulates action that can be taken by an agent.

    Action has the form (move, rotation). Move has the form (m_1, m_2) where:
        m_1 == -1 - move left by 1
        m_1 ==  1 - move right by 1
        m_2 == 1 - move up by 1
        m_2 ==  -1 - move down by 1

    Rotation is represented in degrees (can be negative) and rotation is performed anti-clockwise.
    """

    def __init__(self, move, rotation):
        """ Initializes action.

        :param move: move to be taken be the agent.
        :param rotation: specifies the number of degrees that agent rotates (can be negative).
        """
        self.move = move
        self.rotation = rotation

    def perform(self, agent_position_and_rotation):
        """Performs the action given the agent's current position and rotation.

        Rotation is performed before moving, since it makes more sense to consider rotation followed by moving as one
        action, compared to moving and then rotating.

        :param agent_position_and_rotation: agent's position and rotation, represented by PositionAndRotation object.
        :return: agent's position and rotation after action, represented by PositionAndRotation object.
        """
        agent_position_and_rotation = self.__perform_rotation(
            agent_position_and_rotation)
        return self.__perform_move(agent_position_and_rotation)

    def __perform_rotation(self, agent_position_and_rotation):
        x, y = agent_position_and_rotation.get_pos()
        agent_rotation = agent_position_and_rotation.rotation
        return PositionAndRotation(x, y,
                                   (agent_rotation + self.rotation) % 360)

    def __perform_move(self, agent_position_and_rotation):
        x, y = agent_position_and_rotation.get_pos()
        agent_rotation = agent_position_and_rotation.rotation

        # To determine new coordinates after the move we need to take into the account the current rotation of the agent.
        if agent_rotation == 0:
            x += self.move[1]
            y -= self.move[0]
        if agent_rotation == 90:
            x += self.move[0]
            y += self.move[1]
        if agent_rotation == 180:
            x -= self.move[1]
            y += self.move[0]
        if agent_rotation == 270:
            x -= self.move[0]
            y -= self.move[1]

        return PositionAndRotation(x, y, agent_rotation)
