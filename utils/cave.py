def get_all_vertices(pickets, shoots):
    vertices = pickets.copy()

    for picket in shoots:
        for vt in picket:
            vertices.append(vt)

    return vertices


def get_points_around(point, step):
    x = point[0]
    y = point[1]
    z = point[2]

    points = [
        [x+step, y+step, z+step],
        [x+step, y+step, z-step],
        [x+step, y-step, z+step],
        [x+step, y-step, z-step],
        [x-step, y+step, z+step],
        [x-step, y+step, z-step],
        [x-step, y-step, z+step],
        [x-step, y-step, z-step],
    ]
    
    return points
