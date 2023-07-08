import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import csr_matrix, lil_matrix, diags, eye
from scipy.sparse.linalg import eigs

def adjacency_matrix_sparse(triangles, num_vertices = None):

    if num_vertices is None:
        num_vertices = triangles.max() + 1
        
    A = lil_matrix((num_vertices, num_vertices), dtype=np.uint32)

    for t in triangles:
        v1,v2,v3 = t
        A[v1,v2] = 1
        A[v2,v3] = 1
        A[v3,v1] = 1
        A[v2,v1] = 1
        A[v3,v2] = 1
        A[v1,v3] = 1

    A = A.tocsr()
    return A


def adjacency_matrix_dense(triangles, num_vertices=None):

    if num_vertices is None:
        num_vertices = triangles.max() + 1
        
    A = np.zeros((num_vertices, num_vertices))

    for t in triangles:
        v1,v2,v3 = t
        A[v1,v2] = 1
        A[v2,v3] = 1
        A[v3,v1] = 1
        A[v2,v1] = 1
        A[v3,v2] = 1
        A[v1,v3] = 1

    return A


def degree_matrix(adj, exponent=1):

    num_vertices = adj.shape[0]
    diagonals = np.zeros(num_vertices)

    if exponent==1:
        for i in range(num_vertices):
            diagonals[i] = adj[i,:].toarray().sum()
        return diags(diagonals, format="csr", dtype=np.int32)
    else:
        for i in range(num_vertices):
            diagonals[i] = adj[i,:].toarray().sum().astype(np.float32)**exponent
        return diags(diagonals, format="csr", dtype=np.float32)


def random_walk_laplacian(triangles):

    num_vertices = triangles.max() + 1
    I = eye(num_vertices, num_vertices, 0)
    A = adjacency_matrix_sparse(triangles, num_vertices)
    Dinv = degree_matrix(A, exponent=1)

    L = I - Dinv @ A

    return L


def delta_coordinates(vertices, triangles):

    L = random_walk_laplacian(triangles)

    return L @ vertices

def sample_colormap(scalars):

    colormap = cm.get_cmap("cividis")
    colors = colormap(scalars)

    return colors[:,:-1]


# ------------------------SIGNATURES-------------------------------
def curvature(mesh, vertices, triangles):
    # calculate delta coordinates
    delta = delta_coordinates(vertices, triangles)
    # calculate norm of delta vector
    norm = np.sqrt((delta * delta).sum(-1))
    norm = (norm - norm.min()) / (norm.max() - norm.min())
    return norm



def saliency(mesh, vertices, triangles):

    # get adjacency matrix
    adjacency_matrix = adjacency_matrix_sparse(triangles)

    # compute vertex saliency
    vertex_saliency = []
    for vertex_id in range(len(vertices)):
        saliency = 0.0

        neighbor_indices = adjacency_matrix[vertex_id].nonzero()[1]
        for neighbor_id in neighbor_indices:
            neighbor_normal = mesh.vertex_normals[neighbor_id]
            vertex_normal = mesh.vertex_normals[vertex_id]
            angle = np.arccos(np.dot(neighbor_normal, vertex_normal))
            saliency += angle 
        vertex_saliency.append(saliency)

    # normalize
    max_saliency = max(vertex_saliency)
    min_saliency = min(vertex_saliency)
    normalized_saliency = [(s - min_saliency) / (max_saliency - min_saliency) for s in vertex_saliency]
    return normalized_saliency
