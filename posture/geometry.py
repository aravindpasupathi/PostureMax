import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_angle_three_points(a: 'Point', b: 'Point', c: 'Point') -> float:
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_degrees = np.degrees(np.arccos(cosine_angle))
    return float(angle_degrees)
