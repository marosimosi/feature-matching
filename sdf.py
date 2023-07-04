import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from math import sqrt

MODEL = "barbara_remeshed"
# MODEL = "ioulia"
# MODEL = "cat"
# MODEL = "camel"

# MODEL = "deformed"
# MODEL = "lookup"

# MODEL = "horse"
# MODEL = "pose2"
# MODEL = "gallop1"



def calculate_sdf(vertices, triangles, vertex_normals, triangle_normals, points):
    angle_limit = np.cos(np.radians(45))
    sdf_values = np.zeros(points.shape[0], dtype=float)  
    epsilon = 1e-6  # avoid division by zero
    
    for i, vertex in enumerate(points):
        total_distance = 0.0
        total_triangles = 0
        for k in range(triangles.shape[0]):
            triangle = triangles[k]
            triangle_normal = triangle_normals[k]

            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]
            
            # Check if the vertex is on the same side as the triangle normal
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





""" # Apply clustering based on SDF values
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(sdf_values.reshape(-1, 1))
clusters = kmeans.labels_

# print the cluster centers
print(f"Cluster centers: {kmeans.cluster_centers_}")



# Visualize the clusters
colors = np.zeros((len(points), 3))
for i, cluster_label in enumerate(clusters-1):
    if cluster_label == 0:
        colors[i] = [1, 0, 0]   # Red
    elif cluster_label == 1:
        colors[i] = [0, 1, 0]   # Green
    elif cluster_label == 2:
        colors[i] = [0, 0, 1]   # Blue
    elif cluster_label == 3:
        colors[i] = [0.5, 0.5, 0]   # Yellow 
    # elif cluster_label == 4:
    #     colors[i] = [0.5, 0.5, 0.5]   # Gray

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd]) """


