import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
import kd_tree


MODEL = "barbara_remeshed"
# MODEL = "ioulia"
# MODEL = "cat"
# MODEL = "camel"

# MODEL = "deformed"
# MODEL = "lookup"

# MODEL = "horse"
# MODEL = "pose2"
# MODEL = "gallop1"


# -------------------------LOAD DATA
mesh = o3d.io.read_triangle_mesh("models/" + MODEL + ".obj")

pcd = o3d.io.read_point_cloud("pcd/"+MODEL+"_PCD.ply")
points = np.asarray(pcd.points)

sdf_values = np.load("sdf/"+MODEL+"_SDF.npy")



# --------------------------CLUSTER THE POINTS
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
    elif cluster_label == 4:
        colors[i] = [0.5, 0.5, 0.5]   # Gray

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])



# ------------------------------SEGMENT THE MESH

# Get the indices of the sampled points in the mesh.vertices array
kdtree = o3d.geometry.KDTreeFlann(mesh)
indices = np.zeros(len(points))

for i in range(len(points)):
    _, idx, _ = kdtree.search_knn_vector_3d(points[i], 1)
    idx = int(idx[0])
    indices[i] = idx

indices = np.asarray(indices, dtype=np.int32)

# Assign the cluster labels to the mesh vertices
mesh_colors = np.ones((len(mesh.vertices), 3))
for i in range(len(indices)):
    mesh_colors[indices[i]] = colors[i]
mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)


# Visualize the mesh
o3d.visualization.draw_geometries([mesh])



