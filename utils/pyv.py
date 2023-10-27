import pyvista as pv
from scipy import spatial


def points_to_convexhull_mesh(picket_points, comp_normals=True, subd=True):
    points = picket_points[spatial.ConvexHull(picket_points).vertices]
    faces = spatial.ConvexHull(points).simplices

    mesh = pv.PolyData.from_regular_faces(points, faces)
    
    if subd:
        mesh.subdivide(1, inplace=True)
    if comp_normals:
        mesh.compute_normals(auto_orient_normals=True, inplace=True)
    
    return mesh
