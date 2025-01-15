 
import torch
import torch.nn as nn
import numpy as np

device = 'cuda'
class FaceNormals(nn.Module):
    def __init__(self, normalize=True):
        super(FaceNormals, self).__init__()
        self.normalize = normalize

    def forward(self, vertices, faces):
        v = vertices
        f = faces
        vertices = vertices.squeeze(0)
        # if v.dim() == (f.dim() + 1):
        #     f = f.unsqueeze(0).repeat(v.shape[0], 1, 1)

        # triangles = torch.gather(v, 1, f.unsqueeze(-1).repeat(1, 1, v.shape[-1]))
        # 
        full_vertices = vertices[faces] #F,C=3,3
        v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
        face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
        if self.normalize:
            face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
        return face_normals #F,3

def get_shape_matrix(x):
    
    # 
    if x.dim() == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif x.dim() == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError

def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)
    return Ds @ Dm_inv

def green_strain_tensor(F):
    I = torch.eye(2, dtype=F.dtype, device=F.device)
    Ft = F.transpose(-1, -2)
    return 0.5 * (Ft @ F - I)

def get_edge_length(vertices, edges):
    
    v0 = torch.gather(vertices, 1, edges[:, 0].unsqueeze(-1).repeat(1, 1, vertices.shape[-1]))
    v1 = torch.gather(vertices, 1, edges[:, 1].unsqueeze(-1).repeat(1, 1, vertices.shape[-1]))
    
    return torch.norm(v0 - v1, dim=-1)

def gather_triangles(vertices, indices):
    # 
    if vertices.dim() == (indices.dim() + 1):
        indices = indices.unsqueeze(0).expand(vertices.size(0), -1, -1)

    # vertices = vertices[:,:2]
    # 
    # indices = indices.unsqueeze(-1)
    # batch_dims = vertices.dim() - 2
    # triangles = torch.gather(vertices, -2, 
    #                          indices.unsqueeze(-1).repeat(1,1,3))
    # 
    triangles = vertices[indices]
    # 
    return triangles
def get_face_areas(vertices, faces):

    if torch.is_tensor(vertices):
        vertices = vertices.cpu().numpy()

    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    return np.linalg.norm(np.cross(u, v), axis=-1) / 2.0


def get_face_areas_batch(vertices, faces):
    
    # v0 = torch.gather(vertices, 1, faces[:, 0].unsqueeze(-1).repeat(1, 1, vertices.shape[-1]))
    # v1 = torch.gather(vertices, 1, faces[:, 1].unsqueeze(-1).repeat(1, 1, vertices.shape[-1]))
    # v2 = torch.gather(vertices, 1, faces[:, 2].unsqueeze(-1).repeat(1, 1, vertices.shape[-1]))

    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    
    u = v2 - v0
    v = v1 - v0

    return torch.norm(torch.cross(u, v, dim=-1), dim=-1) / 2.0


def get_vertex_mass(vertices, faces, density, dtype=torch.float32):
    """
        Computes the mass of each vertex according to triangle areas and fabric density
    """
    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:, 0], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 1], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 2], triangle_masses / 3)

    return torch.from_numpy(vertex_masses).to(dtype)


def get_vertex_connectivity(faces, dtype=torch.int32):
    """
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects
    """
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    return torch.tensor(list(edges), dtype=dtype).to(device)


def get_face_connectivity(faces, dtype=torch.int32):
    """
    Returns a list of adjacent face pairs
    """
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()

    edges = get_vertex_connectivity(faces).cpu().numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e].append(i)

    adjacent_faces = []
    for key in G:
        assert len(G[key]) < 3
        if len(G[key]) == 2:
            adjacent_faces.append(G[key])

    return torch.tensor(adjacent_faces, dtype=dtype).to(device)

def get_face_connectivity_edges(faces, dtype=torch.int32):
    """
    Returns a list of edges that connect two faces
    (i.e., all the edges except borders)
    """
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()  # Convert tensor to numpy for set operations

    edges = get_vertex_connectivity(faces).cpu().numpy()

    G = {tuple(e): [] for e in edges}
    
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))  # Sort vertices in the edge to avoid duplicates
            G[e].append(i)

    adjacent_face_edges = []
    
    for key in G:
        assert len(G[key]) < 3
        if len(G[key]) == 2:
            adjacent_face_edges.append(list(key))

    return torch.tensor(adjacent_face_edges, dtype=dtype).to(device)


def get_vertex_connectivity(faces, dtype=torch.int32):
    """
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects.
    """
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()  # Convert tensor to numpy array for processing.

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices  # Get adjacent vertex index
            edges.add(tuple(sorted([f[i], f[j]])))  # Add unique edge

    return torch.tensor(list(edges), dtype=dtype).to(device)


def get_edge_length(vertices, edges):
    edges = edges.to(torch.int64)
    v0 = torch.gather(vertices, 0, edges[:, 0].unsqueeze(-1).expand(-1, vertices.shape[-1]))
    v1 = torch.gather(vertices, 0, edges[:, 1].unsqueeze(-1).expand(-1, vertices.shape[-1]))
    
    return torch.norm(v0 - v1, dim=-1)

def get_v_connect_dict(vertex_connectivity):
    vertex_connectivity = vertex_connectivity.cpu().numpy()
    vertex_dict = {}

    # Iterate over each row in the vertex connectivity tensor
    for i in range(vertex_connectivity.shape[0]):
        vertex1, vertex2 = vertex_connectivity[i]

        # Add vertex2 to the connected vertex IDs of vertex1
        if vertex1 in vertex_dict:
            vertex_dict[vertex1].append(vertex2)
        else:
            vertex_dict[vertex1] = [vertex2]

        # Add vertex1 to the connected vertex IDs of vertex2
        if vertex2 in vertex_dict:
            vertex_dict[vertex2].append(vertex1)
        else:
            vertex_dict[vertex2] = [vertex1]
    return vertex_dict


def dict_to_ragged_tensor(dic):
    values = []
    row_splits = [0]
    for key in sorted(dic.keys()):
        values.extend(dic[key])
        row_splits.append(row_splits[-1] + len(dic[key]))
    ragged = tf.RaggedTensor.from_row_splits(values, row_splits)
    return ragged


def isometry_covar(v, ragged):
    """
     compute covariance matrix.
        1/K * (X_mean-Xn)'@(X_mean-Xn)
    """
    neigh = tf.gather(v, ragged, axis=1)
    diff = (neigh - tf.reduce_mean(neigh, axis=-2, keepdims=True)).to_tensor()
    num_neigh = tf.cast(neigh.row_lengths(axis=2), diff.dtype).to_tensor()
    covar = torch.matmul(diff, diff, transpose_a=True) / tf.expand_dims(tf.expand_dims(num_neigh, -1), -1)
    return covar
