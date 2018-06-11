class State(object):
    def __init__(self, x, y, theta, label, embedding, target_eq_obs, target, state_id, angle, r,
                 target_angle, target_distance):

        self.x = x
        self.y = y
        self.theta = theta
        self.label = label
        self.embedding = embedding
        self.target_eq_obs = target_eq_obs
        self.target = target
        self.state_id = state_id
        # Polar coordinates
        self.angle = angle
        self.r = r
        # Angle and distance from the target
        self.target_angle = target_angle
        self.target_distance = target_distance

    def __eq__(self, other):
        return self.x == other.x and \
               self.y == other.y and \
               self.theta == other.theta and \
               self.label == other.label and\
               self.embedding == other.embedding and \
               self.target_eq_obs == other.target_eq_obs and \
               self.target == other.target and \
               self.state_id == other.state_id and \
               self.angle == other.angle and \
               self.r == other.r and \
               self.target_angle == other.target_angle and \
               self.target_distance == other.target_distance
