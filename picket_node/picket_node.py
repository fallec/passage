import math
from dataclasses import dataclass, field

from vector3d.vector3d import Vector3D
from custom_math.coordinates import decart_to_eiler


@dataclass
class PicketNode:
    """
    Base Picket class
    """
    name:str
    coordinates: Vector3D = Vector3D()
    prev: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    shoots_raw: list[Vector3D] = field(default_factory=list)
    shoots: list[Vector3D] = field(default_factory=list)
    length: float = 0
    width: float = 0
    height: float = 0
    l_to_current: list[float] = field(default_factory=list)
    Az_to_current: list[float] = field(default_factory=list)
    An_to_current: list[float] = field(default_factory=list)
    l_from_current: list[float] = field(default_factory=list)
    Az_from_current: list[float] = field(default_factory=list)
    An_from_current: list[float] = field(default_factory=list)
    An_ring: float = 0
    Az_ring: float = 0
    rings: list[list[Vector3D]] = field(default_factory=list)

    def set_coordinates(self, vector: Vector3D) -> None:
        self.coordinates = vector
    
    def add_coordinates(self, vector: Vector3D) -> None:
        self.coordinates += vector
    
    def get_coordinates(self) -> Vector3D:
        return self.coordinates
    
    def set_way_from_current(self, l: float, Az: float, An: float) -> None:
        self.l_from_current.append(l)
        self.Az_from_current.append(Az)
        self.An_from_current.append(An)
    
    def set_way_to_current(self, l: float, Az: float, An: float) -> None:
        self.l_to_current.append(l)
        self.Az_to_current.append(Az)
        self.An_to_current.append(An)
    
    def add_shoot(self, vector: Vector3D) -> None:
        _ = self.shoots_raw.append(vector)
    
    def get_distance_to(self, picket: 'PicketNode'):
        return self.coordinates.get_distance_to(picket.coordinates)
    
    def fix_stats(self, box_size: float = 1.) -> None:
        self.shoots: list[Vector3D] = []
        for shoot in self.shoots_raw:
            self.shoots.append(self.coordinates + shoot)
        if not self.shoots:
            self.length = box_size
            self.width = box_size
            self.height = box_size
        else:
            self.length = abs(max(shoot.X for shoot in self.shoots) - min(shoot.X for shoot in self.shoots))
            self.width = abs(max(shoot.Y for shoot in self.shoots) - min(shoot.Y for shoot in self.shoots))
            self.height = abs(max(shoot.Z for shoot in self.shoots) - min(shoot.Z for shoot in self.shoots))
            self.length = max(self.length, box_size)
            self.width = max(self.width, box_size)
            self.height = max(self.height, box_size)
    
    def set_ring(self, prev: Vector3D, next: Vector3D) -> None:
        vec: Vector3D = next - prev
        _, Az, An = decart_to_eiler(*vec.to_array())
        self.An_ring = An
        self.Az_ring = Az
        
        print(prev, next, vec)
        print(self.Az_ring, self.An_ring)
    
    def __repr__(self) -> str:
        return (f'\n{self.name}: {self.coordinates}'
                f'\n{self.prev} {self.next}'
                f'\nshoots: {len(self.shoots)}')
    
    def __str__(self) -> str:
        return (f'\n{self.name}: {self.coordinates}'
                f'\n{self.prev} {self.next}'
                f'\nshoots: {len(self.shoots)}')
