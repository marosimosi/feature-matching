import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import utility as U


# Load the point cloud 
pcd = o3d.io.read_point_cloud("barbara.ply")

# normals needed for BPA
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Ball Pivoting Algorithm with multiple radii
radii = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([mesh])

# save the mesh
o3d.io.write_triangle_mesh("barbara.obj", mesh)

# for later use
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()



# -----------CURVATURE----------------

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
# o3d.visualization.draw_geometries([mesh])       # !!!  CTRL + 9 to show normals, CTRL + 1 to return to curvature



# -----------SALIENCY------------------

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

