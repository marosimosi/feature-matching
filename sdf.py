import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d

def calculate_sdf(vertices, triangles, vertex_normals, points):
    sdf_values = np.zeros(vertices.shape[0], dtype=float)  # Specify float dtype for sdf_values
    epsilon = 1e-6  # Small value to avoid division by zero
    
    for i, vertex in enumerate(points):
        total_distance = 0.0
        for triangle in triangles:
            v0 = vertices[triangle[0]]
            v1 = vertices[triangle[1]]
            v2 = vertices[triangle[2]]
            
            n0 = vertex_normals[triangle[0]]
            n1 = vertex_normals[triangle[1]]
            n2 = vertex_normals[triangle[2]]
            
            # Calculate the triangle normal
            triangle_normal = np.cross(v1 - v0, v2 - v0).astype(float)
            triangle_normal /= np.linalg.norm(triangle_normal)
            
            # Check if the vertex is on the same side as the triangle normal
            if np.dot(triangle_normal, vertex - v0) <= 0.0:
                continue
            
            # Calculate the cone direction
            cone_direction = triangle_normal if np.dot(triangle_normal, vertex) >= 0.0 else -triangle_normal
            
            # Send rays inside the cone and calculate distances
            ray_distances = []
            for j in range(3):
                v = vertices[triangle[j]]
                n = vertex_normals[triangle[j]]
                
                # Calculate the ray direction inside the cone
                ray_direction = np.cross(n, cone_direction)
                ray_direction /= np.linalg.norm(ray_direction)
                
                # Calculate the ray distance to the other side of the mesh
                ray_distance = np.dot(ray_direction, v - vertex) / (np.dot(ray_direction, triangle_normal) + epsilon)
                ray_distances.append(ray_distance)
            
            # Calculate the weighted sum of ray distances
            total_distance += max(ray_distances)
        
        sdf_values[i] = total_distance
    
    return sdf_values




mesh = o3d.io.read_triangle_mesh("barbara_remeshed.obj")
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
vertex_normals = np.asarray(mesh.vertex_normals)


# Sample points uniformly for SDF
pcd = mesh.sample_points_uniformly(number_of_points=100)
points = np.asarray(pcd.points)

# Calculate the SDF values for each point
sdf_values = calculate_sdf(vertices, triangles, vertex_normals, points)

print(f"SDF values: {sdf_values}")



# Apply clustering based on SDF values
num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(sdf_values.reshape(-1, 1))
clusters = kmeans.labels_

print(f"Cluster assignments: {clusters}")



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
        colors[i] = [1, 1, 0]   # Yellow 
    elif cluster_label == 4:
        colors[i] = [1, 0, 1]   # Magenta

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])


