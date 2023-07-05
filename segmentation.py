import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from scipy.sparse import csgraph
import utility as U

MODEL = "barbara_remeshed"


# MODEL = "human1"
# MODEL = "human2"

# MODEL = "lion1"
# MODEL = "lion2"
# MODEL = "lion3"
# MODEL = "lion4"
# MODEL = "lion5"

# MODEL ="cat1"
# MODEL = "cat2"

MODEL = "armadillo"


# MODEL = "deformed"
# MODEL = "lookup"

# MODEL = "gallop"
# MODEL = "gallop1"
# MODEL = "gallop2"
# MODEL = "gallop4"




# -------------------------LOAD DATA
mesh = o3d.io.read_triangle_mesh("models/" + MODEL + ".obj")

pcd = o3d.io.read_point_cloud("pcd/"+MODEL+"_PCD.ply")
points = np.asarray(pcd.points)

sdf_values = np.load("sdf/"+MODEL+"_SDF.npy")



# --------------------------CLUSTER THE POINTS
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
kmeans.fit(sdf_values.reshape(-1, 1))
clusters = kmeans.labels_

# Get the indices that sort the cluster centers in ascending order
sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())

# Rearrange the cluster labels based on the sorted indices
reordered_clusters = np.zeros_like(clusters)
for i, cluster_id in enumerate(sorted_indices):
    reordered_clusters[clusters == cluster_id] = i

clusters = reordered_clusters

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
# o3d.visualization.draw_geometries([pcd])



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
mesh_clusters = np.ones(len(mesh.vertices))
for i in range(len(indices)):
    mesh_clusters[indices[i]] = clusters[i]


vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

num_vertices = len(vertices)
new_clusters = np.zeros(num_vertices)


# Compute adjacency matrix and find the shortest path distances from each vertex to the clustered points
A = U.adjacency_matrix_sparse(triangles, num_vertices)
distances = csgraph.dijkstra(A, indices=indices, directed=False, unweighted=True, min_only=False)

for i in range(num_vertices):
    if i in indices:
        # Vertex already belongs to a cluster
        cluster_index = np.where(indices == i)[0][0]
        new_clusters[i] = clusters[cluster_index]

    else:
        # Find the shortest path distances to all clustered points
        v_distances = distances[:, i]

        # Sort the distances and get the indices of the 5 closest clustered points
        closest_indices = np.argsort(v_distances)[:5]
        closest_clusters = clusters[closest_indices]

        # Count the occurrences of each cluster label
        label_counts = np.bincount(closest_clusters)

        # Sort the labels by count in descending order
        sorted_labels = np.argsort(label_counts)[::-1]

        # Assign the majority cluster label to the vertex
        new_clusters[i] = sorted_labels[0]

for i in range(len(indices)):
    v_distances = distances[i, :]
    closest_indices = np.argsort(v_distances)[:10]
    closest_clusters = new_clusters[closest_indices]
    label_counts = np.bincount(closest_clusters.astype(np.int32))
    sorted_labels = np.argsort(label_counts)[::-1]

    new_clusters[indices[i]] = sorted_labels[0]



mesh_colors = np.zeros((len(vertices), 3))
for i, cluster_label in enumerate(new_clusters-1):
    if cluster_label == 0:
        mesh_colors[i] = [1, 0, 0]   # Red
    elif cluster_label == 1:
        mesh_colors[i] = [0, 1, 0]   # Green
    elif cluster_label == 2:
        mesh_colors[i] = [0, 0, 1]   # Blue

mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
o3d.visualization.draw_geometries([mesh])
