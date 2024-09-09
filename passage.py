import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from vector3d.vector3d import Vector3D
from picket_node.picket_node import PicketNode
from custom_math.coordinates import eiler_to_decart


class CaveBase:
    """
    Base cave class
    """
    def __decode_line(self, line: str)  -> Optional[list[str, float]]:
        arr = line.split('\t')
        if len(arr) >= 5:
            current, target, l, phi, teta = arr[0:5]
            if phi == '-': phi = 0
            if teta == '-': teta = 0
            try:
                l = float(l)
                phi = math.radians(- float(phi))
                teta = math.radians(float(teta))
            except ValueError:
                return None
            return current, target, l, phi, teta
        else:
            return None

    def __proceed_pline(self, pline: list[str]) -> None:
        current, target, l, Az, An = pline
        current: PicketNode = self._pickets.get(current, PicketNode(current))
        x, y, z = eiler_to_decart(l, Az, An)
        if target == '-':
            _ = current.add_shoot(Vector3D(x, y, z))
        else:
            target: PicketNode = self._pickets.get(target, PicketNode(target))
            _ = target.set_coordinates(Vector3D(x, y, z))
            _ = target.set_way_to_current(l, Az, An)
            _ = current.set_way_from_current(l, Az, An)
            _ = current.next.append(target.name)
            _ = target.prev.append(current.name)
            self._pickets[target.name] = target
        self._pickets[current.name] = current
    
    def fix_cave(self):
        length = 0.
        fixed = []
        queue = [self.startname]
        while queue:
            current_name = queue.pop()
            current: PicketNode = self._pickets[current_name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
                length += sum(
                    current.get_distance_to(
                        self._pickets[item]) if not item in fixed else 0 for item in current.next)
            if current.prev:
                for item in current.prev:
                    if item in fixed:
                        prev: PicketNode = self._pickets[item]
                        current.add_coordinates(prev.get_coordinates())
                        break
            fixed.append(current_name)
        self.length = length
        self.picket_count = len(fixed)
        
    def fix_pickets(self, box_size: float = 1.):
        fixed = []
        queue = [self.startname]
        while queue:
            current_name = queue.pop()
            current: PicketNode = self._pickets[current_name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
            current.fix_stats(box_size)
            prev = self._pickets[current.prev[0]].coordinates if current.prev else current.coordinates
            next = self._pickets[current.next[0]].coordinates if current.next else current.coordinates
            if len(current.next) == 2:
                next += self._pickets[current.next[1]].coordinates
                next /= 2
            current.set_ring(prev, next)
            fixed.append(current_name)
    
    def __init__(self, filename: str, startname: str, box_size: float = 1.) -> None:
        self.filename = filename
        self.startname = startname
        self._pickets: dict[PicketNode] = dict()
        self.length: float = 0
        self.picket_count: int = 0

        ext = filename.split('.')[-1]
        if ext in ['cav', 'CAV', 'txt']:
            with open(filename, 'r') as f:
                data = f.readlines()
                for line in data:
                    pline = self.__decode_line(line)
                    if pline is not None:
                        _ = self.__proceed_pline(pline)                                
        else:
            raise NotImplementedError('Unknown file format')
        
        if not startname in self._pickets.keys():
            raise ValueError('Start name does not exist')
        
        self.fix_cave()
        self.fix_pickets(box_size)
    
    def get_pickets(self) -> dict[PicketNode]:
        return self._pickets
    
    def print_pickets(self):
        fixed = []
        queue = [self.startname]
        while queue:
            name = queue.pop()
            current: PicketNode = self._pickets[name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
            fixed.append(name)
            print(current)
    
    def plot_raw_points(self):
        fixed = []
        queue = [self.startname]
        while queue:
            name = queue.pop()
            current: PicketNode = self._pickets[name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
            fixed.append(name)
        pickets, way = [], []
        Xp, Yp, Zp = [], [], []
        Xw, Yw, Zw = [], [], []
        for item in fixed:
            way.append(self._pickets[item].coordinates)
            for item_ in self._pickets[item].shoots_raw:
                pickets.append(item_ + self._pickets[item].coordinates)
        for item in pickets:
            Xp.append(item.X)
            Yp.append(item.Y)
            Zp.append(item.Z)
        for item in way:
            Xw.append(item.X)
            Yw.append(item.Y)
            Zw.append(item.Z)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(Xp, Yp, Zp, color='black')
        ax.plot(Xw, Yw, Zw, color='red')
        lim = [min(min(Xp), min(Yp), min(Zp)), max(max(Xp), max(Yp), max(Zp))]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()
    
    def __repr__(self) -> str:
        return f'Cave: {self.filename}'
    
    def __str__(self) -> str:
        return f'Cave: {self.filename}'


class Cave(CaveBase):
    """
    Cave class for round shell triangulation
    """
    def __init__(self, filename: str, startname: str, box_size: float = 1.,
                 n_ring_pow: int = 3, m_per_seg: float = 5, ring_coef: float = 0.5) -> None:
        super().__init__(filename, startname, box_size)
        self.faces = []
        self.n_ring_points = 2 ** (n_ring_pow + 1)
        self.n_ring_pow = n_ring_pow
        self.m_per_seg = m_per_seg
        self.ring_coef = ring_coef

        self.ring_template: list[Vector3D] = []
        for j in range(self.n_ring_points):
            alpha = j * math.pi / (self.n_ring_points // 2)
            self.ring_template.append(Vector3D(0, math.sin(alpha), math.cos(alpha)))
    
    def __create_ring(self, An: float, Az: float, coords: Vector3D,
                      length: float, width: float, height: float) -> list[Vector3D]:
        ring = []
        rot_vector = Vector3D(0, 1, 0).rotate(Vector3D(0, 0, 1), Az)

        for vector in self.ring_template:
            new_vector = vector.copy()
            new_vector = new_vector.rotate(Vector3D(0, 0, 1), Az)
            new_vector = new_vector.rotate(rot_vector, - An)
            
            new_vector.X *= length * self.ring_coef
            new_vector.Y *= width * self.ring_coef
            new_vector.Z *= height * self.ring_coef

            ring.append(new_vector + coords)
        return ring
    
    def __get_fixed_WH_coef(self, An: float, Az: float, An_prev: float, Az_prev: float):
        x1, y1, z1 = eiler_to_decart(1, Az, An)
        x2, y2, z2 = eiler_to_decart(1, Az_prev, An_prev)
        angle = math.acos(
            (x1 * x2 + y1 * y2 + z1 * z2) / (Vector3D(x1, y1, z1).get_norm() * Vector3D(x2, y2, z2).get_norm()) - 0.01
            )
        return (1 - angle / math.pi) ** 2
        
    
    def create_faces(self):
        self.faces: list[list[Vector3D]] = []
        fixed = []
        queue = [self.startname]
        while queue:
            name = queue.pop()
            current: PicketNode = self._pickets[name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
            fixed.append(name)
            
            if len(current.prev) > 1:
                raise NotImplementedError('More then 1 previous ponts')
            if len(current.next) > 2:
                raise NotImplementedError('More then 1 next ponts')


            count_from = len(current.An_from_current) if current.An_from_current else len(current.An_to_current) 
            count_to = len(current.An_to_current) if current.An_to_current else len(current.An_from_current)
            sum_Az_from = sum(current.Az_from_current) if current.Az_from_current else sum(current.Az_to_current)
            sum_Az_to = sum(current.Az_to_current) if current.Az_to_current else sum(current.Az_from_current)
            sum_An_from = sum(current.An_from_current) if current.An_from_current else sum(current.An_to_current)
            sum_An_to = sum(current.An_to_current) if current.An_to_current else sum(current.An_from_current)
            coef = self.__get_fixed_WH_coef(
                sum_An_from / count_from,
                sum_Az_from / count_from,
                sum_An_to / count_to,
                sum_Az_to/ count_to
            )
            
            if coef < 0.1 and len(current.next) == 1:
                next_: PicketNode = self._pickets.get(current.next[0])
                next_.shoots_raw += current.shoots_raw
                next_.prev = current.prev
                next_.Az_to_current[0] = (current.Az_to_current[0] + next_.Az_to_current[0]) / 2.
                next_.An_to_current[0] = (current.An_to_current[0] + next_.An_to_current[0]) / 2.
                _ = next_.fix_stats()
                continue

            current.rings.append(self.__create_ring(
                current.An_ring, current.Az_ring, current.coordinates, current.length, current.width, current.height
            ))
            
            rings = []
            if not current.prev or current.name == self.startname:
                rings.append([current.coordinates] * self.n_ring_points)
            else:
                prev: PicketNode = self._pickets[current.prev[0]]
                if len(prev.next) == 1:
                    rings.append(prev.rings[0])
                else:
                    if current.coordinates.get_distance_to(prev.rings[0][self.n_ring_points // 4 + 1]) < current.coordinates.get_distance_to(prev.rings[0][self.n_ring_points - self.n_ring_points // 4]):
                        temp = prev.rings[1].copy()
                        temp = temp[::-1]
                        rings.append(prev.rings[0][:self.n_ring_points // 2 + 1] + temp)
                    else:
                        rings.append([prev.rings[0][0]] + prev.rings[1] + prev.rings[0][self.n_ring_points // 2:])

                steps_count = int(current.l_to_current[0] // self.m_per_seg) + 1
                coords = prev.coordinates
                step = (current.coordinates - prev.coordinates) / steps_count
                for i in range(steps_count - 1):
                    coords += step
                    if len(prev.next) == 2 and i < steps_count / 5:
                        continue
                    weight = 1.1437275736 / math.pi * math.atan(10 * ((i + 1) / (steps_count - 1) - 0.5)) + 0.5
                    rings.append(
                        self.__create_ring(
                            prev.An_ring * (1 - (i + 1) / (steps_count - 1)) + current.An_ring * ((i + 1) / (steps_count - 1)),
                            prev.Az_ring * (1 - (i + 1) / (steps_count - 1)) + current.Az_ring * ((i + 1) / (steps_count - 1)),
                            coords,
                            prev.length  * (1 - weight) + current.length * weight,
                            prev.width  * (1 - weight) + current.width * weight,
                            prev.height  * (1 - weight) + current.height * weight
                        )
                    )

            rings.append(current.rings[0])
            
            if not current.next:
                rings.append([current.coordinates] * self.n_ring_points)
            elif len(current.next) == 2:
                temp = current.rings[0][1:self.n_ring_points - self.n_ring_points // 2]
                temp = [vector - current.coordinates for vector in temp]
                temp = [vector.rotate(current.rings[0][0] - current.coordinates, - math.pi / 2) for vector in temp]
                temp = [vector + current.coordinates for vector in temp]
                current.rings.append(temp)

            for i_r in range(len(rings) - 1):
                for i in range(self.n_ring_points):
                    self.faces.append([
                        rings[i_r + 1][(i + 1) % self.n_ring_points],
                        rings[i_r + 1][i],
                        rings[i_r][i]
                    ])
                    self.faces.append([
                        rings[i_r][i],
                        rings[i_r][(i + 1) % self.n_ring_points],
                        rings[i_r + 1][(i + 1) % self.n_ring_points]
                    ])
    
    def plot_picket_rings(self):
        fixed = []
        queue = [self.startname]
        while queue:
            name = queue.pop()
            current: PicketNode = self._pickets[name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
            fixed.append(name)
        pickets, way = [], []
        Xp, Yp, Zp = [], [], []
        Xw, Yw, Zw = [], [], []
        for picket_name in fixed:
            way.append(self._pickets[picket_name].coordinates)
            for ring in self._pickets[picket_name].rings:
                for item_ in ring:
                    pickets.append(item_)
        for item in pickets:
            Xp.append(item.X)
            Yp.append(item.Y)
            Zp.append(item.Z)
        for item in way:
            Xw.append(item.X)
            Yw.append(item.Y)
            Zw.append(item.Z)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(Xp, Yp, Zp, color='black')
        ax.plot(Xw, Yw, Zw, color='red')
        lim = [min(min(Xp), min(Yp), min(Zp)), max(max(Xp), max(Yp), max(Zp))]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    
    def plot_all_rings(self):
        fixed = []
        queue = [self.startname]
        while queue:
            name = queue.pop()
            current: PicketNode = self._pickets[name]
            if current.next:
                _ = [queue.append(item) if not item in fixed else None for item in current.next]
            fixed.append(name)
        pickets, way = [], []
        Xp, Yp, Zp = [], [], []
        Xw, Yw, Zw = [], [], []
        for picket_name in fixed:
            way.append(self._pickets[picket_name].coordinates)
        for face in self.faces:
            for point in face:
                pickets.append(point)
        for item in pickets:
            Xp.append(item.X)
            Yp.append(item.Y)
            Zp.append(item.Z)
        for item in way:
            Xw.append(item.X)
            Yw.append(item.Y)
            Zw.append(item.Z)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(Xp, Yp, Zp, color='black')
        ax.plot(Xw, Yw, Zw, color='red')
        lim = [min(min(Xp), min(Yp), min(Zp)), max(max(Xp), max(Yp), max(Zp))]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    
    def create_stl(self, filename):
        with open(filename, 'wb') as fh:

            def b(s, encoding='ascii', errors='replace'):
                if isinstance(s, str):
                    return bytes(s, encoding, errors)
                else:
                    return s

            def p(s, file):
                file.write(b('%s\n' % s))

            p('solid %s' % 'test', file=fh)

            for row in self.faces:
                norm = np.cross((row[2] - row[0]).to_array(), (row[1] - row[0]).to_array())
                norm_ = np.linalg.norm(norm)
                norm = norm / norm_ if norm_ else norm
                p('facet normal %r %r %r' % tuple(norm), file=fh)
                p('  outer loop', file=fh)
                p('    vertex %r %r %r' % tuple(row[2].to_array()), file=fh)
                p('    vertex %r %r %r' % tuple(row[1].to_array()), file=fh)
                p('    vertex %r %r %r' % tuple(row[0].to_array()), file=fh)
                p('  endloop', file=fh)
                p('endfacet', file=fh)

            p('endsolid %s' % 'test', file=fh)
