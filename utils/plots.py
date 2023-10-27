from matplotlib import pyplot as plt
import pyvista as pv


def plot_vertices(vertices, surface=False, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]
    if surface:
        ax.plot_trisurf(x, y, z, linewidth=0.2)
    else:
        ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_cloud_mesh(cloud, mesh):
    pl = pv.Plotter(shape=(1, 2))
    _ = pl.add_mesh(cloud)
    _ = pl.add_title('Point Cloud')
    pl.subplot(0, 1)
    _ = pl.add_mesh(mesh, color=True, show_edges=True)
    _ = pl.add_title('Passage Mesh')
    pl.show()


def plot_mesh(mesh):
    pl = pv.Plotter()
    _ = pl.add_mesh(mesh, color=True, show_edges=True)
    _ = pl.add_title('Mesh')
    pl.show()
