import numpy as np
import pyvista as pv
from scipy import spatial

from utils import plots, cave, pyv
from test_data import test_pickets, test_shoots


vertices = np.array(cave.get_all_vertices(test_pickets, test_shoots), dtype='float64')

first = np.array(cave.get_points_around(test_pickets[0], 0.5), dtype='float64')
second = np.array(cave.get_points_around(test_pickets[1], 0.6), dtype='float64')
picket_points = np.concatenate((first, second), axis=0)

final_mesh = pyv.points_to_convexhull_mesh(picket_points)

for i in range(len(test_pickets) - 2):
    first = np.array(cave.get_points_around(test_pickets[i + 1], 0.5), dtype='float64')
    second = np.array(cave.get_points_around(test_pickets[i + 2], 0.6), dtype='float64')
    picket_points = np.concatenate((first, second), axis=0)

    mesh = pyv.points_to_convexhull_mesh(picket_points)
    
    final_mesh = final_mesh.boolean_union(mesh)
    final_mesh.compute_normals(auto_orient_normals=True, inplace=True, split_vertices=True)

for i in range(len(test_pickets)):
    picket_points = test_shoots[i]
    picket_points.append(test_pickets[i])
    picket_points = np.array(picket_points, dtype='float64')
    picket_points += np.random.rand(len(picket_points), 3) / 2

    mesh = pyv.points_to_convexhull_mesh(picket_points)

    final_mesh = final_mesh.boolean_union(mesh)
    final_mesh.compute_normals(auto_orient_normals=True, inplace=True, split_vertices=True)

cloud = pv.wrap(vertices)
plots.plot_cloud_mesh(cloud, final_mesh)

final_mesh.save('passage.stl')








# empty_mesh = pv.PolyData()

# cloud = pv.wrap(np.array(vertices))

# faces = []
# i, offset = 0, 0
# cc = mesh.cells # fetch up front
# while i < mesh.n_cells:
#     nn = cc[offset]
#     faces.append(cc[offset+1:offset+1+nn])
#     offset += nn + 1
#     i += 1

# final_mesh.plot_normals(faces=True)
# print(final_mesh.is_all_triangles)
# final_mesh.triangulate(inplace=True)

# mesh = pv.PolyData(mesh.points, np.array(faces))
# print(mesh.is_all_triangles)
# print(mesh)
#mesh = mesh.boolean_union(empty_mesh)

# mesh = cloud.delaunay_3d()
# points = vertices[spatial.ConvexHull(vertices).vertices]
# print(len(points))

# mp_plots.plot_vertices(vertices, surface=False)

# sphere_a = pv.Sphere()
# sphere_a.compute_normals(auto_orient_normals=True, inplace=True)
# sphere_a.plot_normals(faces=True)
# sphere_b = pv.Sphere(center=(0.5, 0, 0))
# sphere_b.compute_normals(auto_orient_normals=True, inplace=True)
# sphere_b.plot_normals(faces=True)
# result = sphere_a.boolean_union(sphere_b)
# result.compute_normals(auto_orient_normals=True, inplace=True)
# result.plot_normals(faces=True,)


# pl = pv.Plotter()
# _ = pl.add_mesh(sphere_a, color='r', style='wireframe', line_width=3)
# _ = pl.add_mesh(sphere_b, color='b', style='wireframe', line_width=3)
# _ = pl.add_mesh(result, color='lightblue')
# _ = pl.add_mesh(result, color='g', style='wireframe', line_width=3)
# pl.camera_position = 'xz'
# pl.show()
