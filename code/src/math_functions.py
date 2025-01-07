import numpy as np


def side_area(x, points, theta_min=0, theta_max=np.pi):
    """
    Compute the points within a specific angular range relative to a reference vector x.

    :param x: A 2D vector defining the reference direction (numpy array).
    :param points: A set of 2D points (numpy array of shape (N, 2)).
    :param theta_min: The minimum angle (in radians) for the range (inclusive).
    :param theta_max: The maximum angle (in radians) for the range (exclusive).
    :return: A subset of points within the specified angular range.
    """

    # Normalize the reference vector
    o_x = o(x)

    # Compute angles of all points relative to the reference vector
    angles = []
    for y in points:
        o_y = o(y)
        dot_product = np.dot(o_x, o_y)
        angle = np.arctan2(o_y[1], o_y[0]) - np.arctan2(o_x[1], o_x[0])  # Angle relative to x
        angle = (angle + 2 * np.pi) % (2 * np.pi)  # Normalize angle to [0, 2π)
        angles.append(angle)

    # Filter points within the specified angle range
    filtered_points = []
    for i, angle in enumerate(angles):
        # if theta_min <= angle < theta_max:
        #     filtered_points.append(points[i])
        if theta_min <= angle < theta_max or (theta_max < theta_min and (angle < theta_max or angle >= theta_min)):
            filtered_points.append(points[i])

    return np.array(filtered_points)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def o(vector):
    """Normalize a 2D vector to unit length."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)

def distance_to_set(x, X):
    """
    Compute the shortest Euclidean distance between a point x and a set of points X.

    :param x: A 2D point (numpy array).
    :param X: A set of 2D points (numpy array of shape (N, 2)).
    :return: The shortest distance between x and the set X.
    """
    distances = np.linalg.norm(X - x, axis=1)  # Compute distances to all points in X
    return np.min(distances)

def rotation_matrix(theta):
    """
    Returns the 2D rotation matrix for a given angle theta.

    Parameters:
        theta (float): The angle in radians to rotate by.

    Returns:
        np.array: A 2x2 rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def C(P):
    '''
    cardinality of the set P
    :param P:
    :return:
    '''
    return len(set(P))

def left_most_right_most_sheep(visible_sheep, source):
    """
    Identify the leftmost and rightmost visible sheep from the perspective of the source.

    :param visible_sheep: Array of positions of visible sheep.
    :param source: The position from which the perspective is taken (e.g., sheepdog or destination).
    :return: Tuple of positions (leftmost_sheep, rightmost_sheep).
    """
    if len(visible_sheep) == 0:
        return None, None

    # Compute relative positions and angles from the source
    relative_positions = visible_sheep - source
    angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])  # Calculate angles

    # Return the positions of the leftmost and rightmost sheep
    leftmost_sheep = visible_sheep[np.argmax(angles)]
    rightmost_sheep = visible_sheep[np.argmin(angles)]

    return leftmost_sheep, rightmost_sheep
