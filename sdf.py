import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from math import sqrt

MODEL = "barbara_remeshed"

# MODEL = "gallop"
# MODEL = "gallop2"
# MODEL = "gallop4"

# MODEL = "crop"
# MODEL = "deformed"
# MODEL = "lookup"

# MODEL = "lion1"
# MODEL = "lion2"
# MODEL = "lion3"
# MODEL = "lion4"

# MODEL ="cat1"
# MODEL = "cat2"

# MODEL = "human2"

# MODEL = "armadillo"





def calculate_sdf(vertices, triangles, vertex_normals, triangle_normals, points):
    angle_limit = np.cos(np.radians(45))
    sdf_values = np.zeros(points.shape[0], dtype=float)  
    
    for i, vertex in enumerate(points):
        total_distance = 0.0
        total_triangles = 0
        for k in range(triangles.shape[0]):
            triangle = triangles[k]
            triangle_normal = triangle_normals[k]

            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]
            
            # Check if the vertex is opposite to the triangle
            if np.dot(vertex_normals[i], -triangle_normal) >= angle_limit:
                continue
            
            # Send rays inside the cone and calculate distances
            ray_distances = []
            for j in range(3):
                v = vertices[triangle[j]]
                n = vertex_normals[triangle[j]]

                ray_distance = sqrt((v[0] - vertex[0])**2 + (v[1] - vertex[1])**2 + (v[2] - vertex[2])**2)
                ray_distances.append(ray_distance)

            
            # Sum the max distance of each triangle
            total_distance += max(ray_distances)
            total_triangles += 1
        
        # Calculate the SDF value for the point
        sdf_values[i] = total_distance / total_triangles
    
    return sdf_values




mesh = o3d.io.read_triangle_mesh("models/" + MODEL + ".obj")

mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
vertex_normals = np.asarray(mesh.vertex_normals)
mesh.compute_triangle_normals()
triangle_normals = np.asarray(mesh.triangle_normals)


# Sample points uniformly for SDF
pcd = mesh.sample_points_uniformly(number_of_points=400)
points = np.asarray(pcd.points)

# Calculate the SDF values for each point
sdf_values = calculate_sdf(vertices, triangles, vertex_normals, triangle_normals, points)

# Normalize the SDF values
sdf_values = (sdf_values - np.min(sdf_values)) / (np.max(sdf_values) - np.min(sdf_values))



# save SDF values
np.save("sdf/" + MODEL + "_SDF.npy", sdf_values)
# save PCD
o3d.io.write_point_cloud("pcd/" + MODEL + "_PCD.ply", pcd)

