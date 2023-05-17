import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Load the point cloud 
pcd = o3d.io.read_point_cloud("ioulia.ply")

# normals needed for BPA
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(10)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# Ball Pivoting Algorithm with multiple radii
radii = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03] #0.01,0.02
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

# visualize the mesh
o3d.visualization.draw_geometries([mesh])

# save the mesh
# o3d.io.write_triangle_mesh("ioulia.obj", mesh)






