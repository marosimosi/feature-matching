import open3d as o3d
import numpy as np

def cut_along_plane(mesh, plane, axis):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_centers = np.mean(vertices[triangles], axis=1)
    keep_indices = np.where(triangle_centers[:, axis] < plane)[0]
    
    mesh_cut = o3d.geometry.TriangleMesh()
    mesh_cut.vertices = o3d.utility.Vector3dVector(vertices)
    
    for index in keep_indices:
        mesh_cut.triangles.append(triangles[index])
    
    return mesh_cut


mesh = o3d.io.read_triangle_mesh("models/barbara_remeshed.obj")

# DEFINE PLANE
plane = 0.35
axis = 2 #z

mesh_cut = cut_along_plane(mesh, plane, axis)
o3d.io.write_triangle_mesh("models/crop.obj", mesh_cut)