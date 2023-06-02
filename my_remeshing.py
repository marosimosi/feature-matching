import open3d as o3d
import numpy as np
import utility as U

def subdivide_mesh(vertices, triangles):
    num_vertices = len(vertices)
    num_triangles = len(triangles)
    
    new_vertices = vertices.copy()
    new_triangles = []
    
    for i in range(num_triangles):
        # Get the indices of the vertices for the current triangle
        v1, v2, v3 = triangles[i]
        
        # Compute the midpoint for each edge
        v12 = (vertices[v1] + vertices[v2]) / 2
        v23 = (vertices[v2] + vertices[v3]) / 2
        v31 = (vertices[v3] + vertices[v1]) / 2
        
        # Add the new vertices to the list
        new_vertices = np.vstack((new_vertices, v12, v23, v31))
        
        # Get the indices of the new vertices
        v12_idx = num_vertices + i * 3
        v23_idx = num_vertices + i * 3 + 1
        v31_idx = num_vertices + i * 3 + 2
        
        # Create the new triangles using the new vertices
        new_triangles.extend([(v1, v12_idx, v31_idx),
                              (v12_idx, v2, v23_idx),
                              (v23_idx, v3, v31_idx),
                              (v12_idx, v23_idx, v31_idx)])
        
    return new_vertices, np.array(new_triangles)


# load initial mesh
mesh = o3d.io.read_triangle_mesh("barbara.obj")
mesh_vertices = np.asarray(mesh.vertices)
mesh_triangles = np.asarray(mesh.triangles)

remesh_vertices, remesh_triangles = subdivide_mesh(mesh_vertices, mesh_triangles)
remesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(remesh_vertices), o3d.utility.Vector3iVector(remesh_triangles))
remesh.compute_vertex_normals()
remesh.compute_triangle_normals()
#save
o3d.io.write_triangle_mesh("my_remesh.obj", remesh)
