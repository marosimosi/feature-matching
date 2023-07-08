import open3d as o3d
import numpy as np
import pymeshlab
import matplotlib.pyplot as plt
import utility as U

def visualize(mesh, signature):
    # colormap
    color_map = plt.cm.viridis
    colors = color_map(signature)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
    # visualize
    o3d.visualization.draw_geometries([mesh])       # !!!  CTRL + 9 to show normals, CTRL + 1 to return



# ---------------------------PCD TO MESH

# Load the point cloud 
pcd = o3d.io.read_point_cloud("models/barbara.ply")
o3d.visualization.draw_geometries([pcd])

# Ball Pivoting Algorithm with multiple radii
radii = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([mesh])

# for later use
mesh_vertices = np.asarray(mesh.vertices)
mesh_triangles = np.asarray(mesh.triangles)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# save the mesh
o3d.io.write_triangle_mesh("models/barbara.obj", mesh)




# --------------------------------ISOTROPIC REMESHING
ms = pymeshlab.MeshSet()

ms.load_new_mesh("models/barbara.obj")
ms.apply_filter("meshing_isotropic_explicit_remeshing")
ms.save_current_mesh("models/barbara_remeshed.obj")

# load the remeshed mesh
remesh = o3d.io.read_triangle_mesh("models/barbara_remeshed.obj")

# for later use
remesh_vertices = np.asarray(remesh.vertices)
remesh_triangles = np.asarray(remesh.triangles)
remesh.compute_vertex_normals()
remesh.compute_triangle_normals()



# -----------------------------------COMPARE
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
o3d.visualization.draw_geometries([remesh], mesh_show_wireframe=True)

mesh_curv = U.curvature(mesh, mesh_vertices, mesh_triangles)
remesh_curv = U.curvature(remesh, remesh_vertices, remesh_triangles)

mesh_sal = U.saliency(mesh, mesh_vertices, mesh_triangles)
remesh_sal = U.saliency(remesh, remesh_vertices, remesh_triangles)


visualize(mesh, mesh_curv)
visualize(remesh, remesh_curv)
visualize(mesh, mesh_sal)
visualize(remesh, remesh_sal)

