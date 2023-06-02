import open3d as o3d
import numpy as np
import pymeshlab
import matplotlib.pyplot as plt
import utility as U
import pymeshlab as ml


# ------------------------SIGNATURES-------------------------------
def curvature(mesh, vertices, triangles):
    # calculate delta coordinates
    delta = U.delta_coordinates(vertices, triangles)
    # calculate norm of delta vector
    norm = np.sqrt((delta * delta).sum(-1))
    norm = (norm - norm.min()) / (norm.max() - norm.min())
    # colormap
    color_map = plt.cm.viridis
    colors = color_map(norm)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
    # visualize
    o3d.visualization.draw_geometries([mesh])       # !!!  CTRL + 9 to show normals, CTRL + 1 to return to curvature


def saliency(mesh, vertices, triangles):
    # compute face areas
    face_areas = []
    for face in triangles:
        p0, p1, p2 = vertices[face]
        v0, v1, v2 = p1 - p0, p2 - p0, p2 - p1
        cross_product = np.cross(v0, v1)
        area = 0.5 * np.linalg.norm(cross_product)
        face_areas.append(area)

    # get adjacency matrix
    adjacency_matrix = U.adjacency_matrix_sparse(triangles)

    # compute vertex saliency
    vertex_saliency = []
    for vertex_id in range(len(vertices)):
        saliency = 0.0

        neighbor_indices = adjacency_matrix[vertex_id].nonzero()[1]
        for neighbor_id in neighbor_indices:
            neighbor_normal = mesh.vertex_normals[neighbor_id]
            vertex_normal = mesh.vertex_normals[vertex_id]
            angle = np.arccos(np.dot(neighbor_normal, vertex_normal))
            saliency += angle * face_areas[neighbor_id]
        vertex_saliency.append(saliency)

    # normalize
    max_saliency = max(vertex_saliency)
    min_saliency = min(vertex_saliency)
    normalized_saliency = [(s - min_saliency) / (max_saliency - min_saliency) for s in vertex_saliency]
    # colormap
    color_map = plt.cm.viridis
    colors = color_map(normalized_saliency)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
    # visualize
    o3d.visualization.draw_geometries([mesh])
# -------------------------------------------------------------------




# Load the point cloud 
pcd = o3d.io.read_point_cloud("barbara.ply")

# Ball Pivoting Algorithm with multiple radii
radii = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([mesh])

# for later use
mesh_vertices = np.asarray(mesh.vertices)
mesh_triangles = np.asarray(mesh.triangles)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# save the mesh
o3d.io.write_triangle_mesh("barbara.obj", mesh)




# ISOTROPIC REMESHING
ms = pymeshlab.MeshSet()

ms.load_new_mesh("barbara.obj")
ms.apply_filter("meshing_isotropic_explicit_remeshing")
ms.save_current_mesh("barbara_remeshed.obj")

# load the remeshed mesh
remesh = o3d.io.read_triangle_mesh("barbara_remeshed.obj")

# for later use
remesh_vertices = np.asarray(remesh.vertices)
remesh_triangles = np.asarray(remesh.triangles)
remesh.compute_vertex_normals()
remesh.compute_triangle_normals()



# COMPARE
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
o3d.visualization.draw_geometries([remesh], mesh_show_wireframe=True)

curvature(mesh, mesh_vertices, mesh_triangles)
curvature(remesh, remesh_vertices, remesh_triangles)

saliency(mesh, mesh_vertices, mesh_triangles)
saliency(remesh, remesh_vertices, remesh_triangles)









