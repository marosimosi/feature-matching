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

    # avail_maps = ["inferno", "magma", "viridis", "cividis"]

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
    # compute face areas
    face_areas = []
    for face in triangles:
        p0, p1, p2 = vertices[face]
        v0, v1, v2 = p1 - p0, p2 - p0, p2 - p1
        cross_product = np.cross(v0, v1)
        area = 0.5 * np.linalg.norm(cross_product)
        face_areas.append(area)

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
            saliency += angle * face_areas[neighbor_id]
        vertex_saliency.append(saliency)

    # normalize
    max_saliency = max(vertex_saliency)
    min_saliency = min(vertex_saliency)
    normalized_saliency = [(s - min_saliency) / (max_saliency - min_saliency) for s in vertex_saliency]
    return normalized_saliency




def concavity(mesh, vertices, triangles):
    # get adjacency matrix
    adjacency_matrix = adjacency_matrix_sparse(triangles)

    vertex_concavity = []
    for vertex_id in range(len(vertices)):
        concavity = 0.0

        neighbor_indices = adjacency_matrix[vertex_id].nonzero()[1]
        for neighbor_id in neighbor_indices:
            neighbor_normal = mesh.vertex_normals[neighbor_id]
            vertex_normal = mesh.vertex_normals[vertex_id]
            angle = np.arccos(np.dot(neighbor_normal, vertex_normal))
            if angle > np.pi / 2:
                concavity += np.pi - angle
        vertex_concavity.append(concavity)

    # normalize
    max_concavity = max(vertex_concavity)
    min_concavity = min(vertex_concavity)
    normalized_concavity = [(s - min_concavity) / (max_concavity - min_concavity) for s in vertex_concavity]
    return normalized_concavity
    
    return concavity


# -------------------------------------------------------------------




""" def have_common_edge(tri1, tri2):

    common = 0
    for i in tri1:
        for j in tri2:
            if i == j:
                common += 1

    if common == 2:
        return True
    else:
        return False


def circumcircle(triangle, vertices):

    v1, v2, v3 = vertices[triangle]

    A = np.array([[v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]],
                  [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]],
                  [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]]])

    b = 0.5 * np.array([np.dot(v2, v2) - np.dot(v1, v1),
                       np.dot(v3, v3) - np.dot(v1, v1),
                       np.dot(v3, v3) - np.dot(v2, v2)])

    try:
        center = np.linalg.solve(A, b)
        radius = np.sqrt(np.dot(v1 - center, v1 - center))
        return center, radius
    except np.linalg.LinAlgError:
        # Handle the case when A is a singular matrix (degenerate triangle)
        return None, None

    return center, radius


def non_common_vertex(tri1, tri2):
    for i in tri1:
        if i not in tri2:
            return i


def flip_edge(tri1, tri2, vertices, triangles):
    v1 = non_common_vertex(tri1, tri2)
    v2 = non_common_vertex(tri2, tri1)

    # v3,v4 are the vertices of the triangles that are not v1 and v2
    v3,v4 = [x for x in tri1 if x != v1 and x != v2]
    
    new_tri1 = [v1, v2, v3]
    new_tri2 = [v1, v2, v4]

    triangles = np.delete(triangles, np.where((triangles == tri1).all(axis=1))[0][0], axis=0)
    triangles = np.delete(triangles, np.where((triangles == tri2).all(axis=1))[0][0], axis=0)
    triangles = np.append(triangles, [new_tri1], axis=0)
    triangles = np.append(triangles, [new_tri2], axis=0)

    return triangles

def collinear(v1, v2, v3):
    if np.linalg.norm(np.cross(v2 - v1, v3 - v1)) == 0:
        return True
    else:
        return False """
