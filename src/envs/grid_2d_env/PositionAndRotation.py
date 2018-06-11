class PositionAndRotation:
    """PositionAndRotation class encapsulates the position and rotation pair."""

    def __init__(self, x, y, rotation):
        """Initialize postion and rotation."""
        self.x = x
        self.y = y
        self.rotation = rotation

    def __eq__(self, other):
        return self.x == other.x and \
               self.y == other.y and \
               self.rotation == other.rotation

    def to_list(self):
        return [self.x, self.y, self.rotation]

    def get_pos(self):
        return self.x, self.y
