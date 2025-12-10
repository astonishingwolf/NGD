from copy import deepcopy
import time
import torch
import torch_scatter
import igl
import scipy as sp
import numpy as np
from scripts.models.remesh.remesh_utils_legacy import *
def calc_sizing_field(verts: torch.Tensor, faces: torch.Tensor, epsilon: float = 0.001, \
                          iters: int = 3, project: bool = False, adaptive: bool = True):

    V = verts.clone().cpu().numpy()
    F = faces.clone().cpu().numpy()
    
    if adaptive:
        
        epsilon_vec = torch.full((V.shape[0],), epsilon, dtype=torch.float64)		
        K = igl.gaussian_curvature(V, F)
        
        m = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        minv = sp.sparse.diags(1 / m.diagonal())
        kn = minv.dot(K)
        
        L = igl.cotmatrix(V, F)
        HN = -minv.dot(L.dot(V))
        # WHY /4 ? Don't know?
        H = np.linalg.norm(HN, axis=1) / 4
        
        # TODO In future implement more robust 
        # Principal curvature implementation 
        delta = np.square(H) - K
        delta[delta < 0] = 0
        
        K_tot = H + np.sqrt(delta)
        
        sizingField = np.sqrt(6 * epsilon_vec / K_tot - 3 * np.square(epsilon_vec))
        
        my_max = 3
        my_min = 0.01
        # breakpoint()
        sizingField[sizingField > my_max] = my_max
        sizingField[sizingField < my_min] = my_min
                
        # Optionally print debugging info
        # print(f"found NaN in sizing field. i: {i}")
        # print(f"ktot: {K_tot[i]}")
        # print(f"epsilon: {epsilon_vec[i]}")
        # print(f"H: {H[i]}")
        # print(f"K: {K[i]}")
        
        sizingField = torch.tensor(sizingField)
        # breakpoint()
        return sizingField
    else:
        return False

@torch.no_grad()
def remesh(
        vertices_etc:torch.Tensor, #V,D
        faces:torch.Tensor, #F,3 long
        min_edgelen:torch.Tensor, #V
        max_edgelen:torch.Tensor, #V
        flip:bool,
        max_vertices=1e6
        ):

    # dummies
    vertices_etc,faces = prepend_dummies(vertices_etc,faces)
    vertices = vertices_etc[:,:3] #V,3
    nan_tensor = torch.tensor([torch.nan],device=min_edgelen.device)
    min_edgelen = torch.concat((nan_tensor,min_edgelen))
    max_edgelen = torch.concat((nan_tensor,max_edgelen))
    # breakpoint()
    # collapse
    edges,face_to_edge = calc_edges(faces) #E,2 F,3
    edge_length = calc_edge_length(vertices,edges) #E
    face_normals = calc_face_normals(vertices,faces,normalize=False) #F,3
    vertex_normals = calc_vertex_normals(vertices,faces,face_normals) #V,3
    face_collapse = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edgelen,area_ratio=0.5)
    ## Have to take the least according to the paper
    shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0) #e[0,1] 0...ok, 1...edgelen=0
    priority = face_collapse.float() + shortness
    # vertices_etc,faces = collapse_edges(vertices_etc,faces,edges,priority)

    # split
    if vertices.shape[0]<max_vertices:
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        vertices = vertices_etc[:,:3] #V,3
        edge_length = calc_edge_length(vertices,edges) #E
        ## Have to take the least according to the paper
        splits = edge_length > max_edgelen[edges].mean(dim=-1)
        vertices_etc,faces = split_edges(vertices_etc,faces,edges,face_to_edge,splits,pack_faces=False)

    vertices_etc,faces = pack(vertices_etc,faces)
    vertices = vertices_etc[:,:3]

    if flip:
        edges,_,edge_to_face = calc_edges(faces,with_edge_to_face=True) #E,2 F,3
        flip_edges(vertices,faces,edges,edge_to_face,with_border=False)

    return remove_dummies(vertices_etc,faces)
    

@torch.no_grad()
def adaptive_remesh(vertices, faces, vertices_orig = None, faces_orig = None, epsilon: float = 0.001, iters: int = 3, project: bool = False, adaptive: bool = True):
    
    """
    Perform adaptive remeshing using Botsch's algorithm.
    
    Args:
        verts: Vertex positions (N x 3)
        faces: Face indices (M x 3)
        epsilon: Target edge length
        iters: Number of remeshing iterations
        project: Whether to project vertices back onto the original surface
        adaptive: Whether to use adaptive sizing field based on curvature
    
    Returns:
        Tuple of (new_vertices, new_faces)
    """
    if vertices_orig is None:
        vertices_orig = vertices
    if faces_orig is None:
        faces_orig = faces

    V = vertices.clone()
    F = faces.clone()
    V0, F0 = V.clone(), F.clone()            

    sizing_field = calc_sizing_field(V, F, epsilon, adaptive)
    high = 0.9 * sizing_field
    low = 0.8 * sizing_field

    # print("Splitting edges...")

    vertices, faces = remesh(vertices_orig, faces_orig, high.to('cuda'), low.to('cuda'), flip = True)
    faces = faces.to('cuda')
    vertices = vertices.to('cuda')
    
    # self._split_vertices_etc()
    # self._vertices.requires_grad_()

    return vertices, faces



