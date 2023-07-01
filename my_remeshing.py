import open3d as o3d
import numpy as np
import utility as U
import matplotlib.pyplot as plt


def visualize(mesh, signature):
    # colormap
    color_map = plt.cm.viridis
    colors = color_map(signature)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
    # visualize
    o3d.visualization.draw_geometries([mesh])       # !!!  CTRL + 9 to show normals, CTRL + 1 to return

def subdivide_mesh(vertices, triangles):
    num_vertices = len(vertices)
    num_triangles = len(triangles)
    
    new_vertices = vertices.copy()
    new_triangles = []
    edge_midpoints = {}  # To store and reuse edge midpoints
    
    # Helper function to get or create a new midpoint vertex
    def get_midpoint_vertex(v_idx1, v_idx2):
        key = tuple(sorted([v_idx1, v_idx2]))
        if key in edge_midpoints:
            return edge_midpoints[key]
        else:
            v = (vertices[v_idx1] + vertices[v_idx2]) / 2
            nonlocal new_vertices  # Use the outer variable new_vertices
            new_vertices = np.vstack((new_vertices, v))
            v_idx = num_vertices + len(edge_midpoints)
            edge_midpoints[key] = v_idx
            return v_idx
    
    for i in range(num_triangles):
        # Get the indices of the vertices for the current triangle
        v1, v2, v3 = triangles[i]
        
        # Compute the midpoint for each edge and get or create the midpoint vertices
        v12_idx = get_midpoint_vertex(v1, v2)
        v23_idx = get_midpoint_vertex(v2, v3)
        v31_idx = get_midpoint_vertex(v3, v1)
        
        # Create the new triangles using the new and existing vertices
        new_triangles.extend([(v1, v12_idx, v31_idx),
                              (v12_idx, v2, v23_idx),
                              (v23_idx, v3, v31_idx),
                              (v12_idx, v23_idx, v31_idx)])
        
    return new_vertices, np.array(new_triangles)




# load initial mesh
mesh = o3d.io.read_triangle_mesh("models/barbara.obj")
mesh_vertices = np.asarray(mesh.vertices)
mesh_triangles = np.asarray(mesh.triangles)

remesh_vertices, remesh_triangles = subdivide_mesh(mesh_vertices, mesh_triangles)
remesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(remesh_vertices), o3d.utility.Vector3iVector(remesh_triangles))
remesh.compute_vertex_normals()
remesh.compute_triangle_normals()
#save
o3d.io.write_triangle_mesh("models/my_remesh.obj", remesh)


# visualize
o3d.visualization.draw_geometries([remesh], mesh_show_wireframe=True)

remesh_curv = U.curvature(remesh, remesh_vertices, remesh_triangles)
remesh_sal = U.saliency(remesh, remesh_vertices, remesh_triangles)

visualize(remesh, remesh_curv)
visualize(remesh, remesh_sal)
