import math
import numpy as np

from custom_math.matrix import get_rotation_matrix


class Vector3D:
    """
    Base Vector 3D class
    """
    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.X = x
        self.Y = y
        self.Z = z
    
    def get_distance_to(self, vector: 'Vector3D') -> float:
        return math.sqrt(
            (self.X - vector.X) ** 2 + (self.Y - vector.Y) ** 2 + (self.Z - vector.Z) ** 2
        )
    
    def get_norm(self):
        return math.sqrt(self.X ** 2 + self.Y ** 2 + self.Z** 2)
    
    def to_array(self):
        return [self.X, self.Y, self.Z]
    
    def rotate(self, axis: 'Vector3D', angle: float):
        return Vector3D(
            *tuple(
                np.dot(get_rotation_matrix(axis.to_array(), angle), self.to_array())
            )
        )
    
    def copy(self):
        return Vector3D(self.X, self.Y, self.Z)
    
    def __radd__(self, other):
        return self.get_norm() + other

    def __add__(self, vector: 'Vector3D'):
        return Vector3D(self.X + vector.X, self.Y + vector.Y, self.Z + vector.Z)

    def __sub__(self, vector: 'Vector3D'):
        return Vector3D(self.X - vector.X, self.Y - vector.Y, self.Z - vector.Z)
    
    def __truediv__(self, div: float):
        return Vector3D(self.X / div, self.Y / div, self.Z / div)
    
    def __repr__(self) -> str:
        return (f'\nVector3D: {round(self.X, 2)} {round(self.Y, 2)} {round(self.Z, 2)}')
    
    def __str__(self) -> str:
        return (f'\nVector3D: {round(self.X, 2)} {round(self.Y, 2)} {round(self.Z, 2)}')
