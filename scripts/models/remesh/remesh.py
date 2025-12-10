from copy import deepcopy
import time
import torch
import torch_scatter
import igl
import scipy as sp
import numpy as np
from scripts.utils.general_utils import *
from scripts.models.remesh.remesh_utils_legacy import *

def loop_subdivision(vertices_orig, faces_orig, face_mask_orig, edge_length_threshold=0.01, device='cuda'):
    face_mask = face_mask_orig.to(torch.bool)
    triangles_to_subdivide = faces_orig[face_mask.squeeze(-1)]
    triangle_indices = torch.where(face_mask.squeeze(-1))[0]
    new_triangles = []
    new_vertices = vertices_orig.tolist()
    midpoint_cache = {}

    def get_edge_length(v1_idx, v2_idx):
        return torch.norm(vertices_orig[v1_idx] - vertices_orig[v2_idx])

    def get_midpoint(v1_idx, v2_idx):
        edge_key = tuple(sorted([v1_idx, v2_idx]))
        
        if edge_key in midpoint_cache:
            return midpoint_cache[edge_key]
        else:
            midpoint = (vertices_orig[v1_idx] + vertices_orig[v2_idx]) / 2.0
            new_vertices.append(midpoint)
            new_idx = len(new_vertices) - 1
            midpoint_cache[edge_key] = new_idx

            return new_idx

    for triangle, tri_index in zip(triangles_to_subdivide,triangle_indices):
        v1, v2, v3 = triangle[0].item(), triangle[1].item(), triangle[2].item()

        if get_edge_length(v1, v2) < edge_length_threshold or get_edge_length(v2, v3) < edge_length_threshold or get_edge_length(v3, v1) < edge_length_threshold:
            face_mask[tri_index] = False
            continue 

        idx_m12 = get_midpoint(v1, v2)
        idx_m23 = get_midpoint(v2, v3)
        idx_m31 = get_midpoint(v3, v1)

        if idx_m12 is None or idx_m23 is None or idx_m31 is None:
            new_triangles.append([v1, v2, v3])
        else:
            new_triangles.append([v1, idx_m12, idx_m31])
            new_triangles.append([idx_m12, v2, idx_m23])
            new_triangles.append([idx_m31, idx_m23, v3])
            new_triangles.append([idx_m12, idx_m23, idx_m31])

    new_triangles = torch.tensor(new_triangles, dtype=torch.long).to(device)
    new_vertices = torch.tensor(new_vertices).to(device)
    vertices_final = new_vertices
    faces_final = torch.cat([faces_orig[~face_mask.squeeze(-1)], new_triangles], dim=0)

    return vertices_final, faces_final, face_mask

def triangle_subdivision(vertices_orig, faces_orig, face_mask_orig, edge_length_threshold=0.01, device='cuda'):
    face_mask = face_mask_orig.to(torch.bool)
    triangles_to_subdivide = faces_orig[face_mask.squeeze(-1)]
    triangle_indices = torch.where(face_mask.squeeze(-1))[0]
    new_triangles = []
    new_vertices = vertices_orig.tolist()
    midpoint_cache = {}

    def get_edge_length(v1_idx, v2_idx):
        return torch.norm(vertices_orig[v1_idx] - vertices_orig[v2_idx])


    for triangle, tri_index in zip(triangles_to_subdivide,triangle_indices):
        v1, v2, v3 = triangle[0].item(), triangle[1].item(), triangle[2].item()

        if get_edge_length(v1, v2) < edge_length_threshold or get_edge_length(v2, v3) < edge_length_threshold or get_edge_length(v3, v1) < edge_length_threshold:
            face_mask[tri_index] = False
            continue 

        centroid = (vertices_orig[v1] + vertices_orig[v2] + vertices_orig[v3]) / 3.0

        new_vertices.append(centroid)
        idx_centroid = len(new_vertices) - 1
        new_triangles.append([v1, v2, idx_centroid])
        new_triangles.append([v2, v3, idx_centroid])
        new_triangles.append([v3, v1, idx_centroid])

    new_triangles = torch.tensor(new_triangles, dtype=torch.long).to(device)
    new_vertices = torch.tensor(new_vertices).to(device)
    vertices_final = new_vertices
    faces_final = torch.cat([faces_orig[~face_mask.squeeze(-1)], new_triangles], dim=0)

    return vertices_final, faces_final, face_mask

@torch.no_grad()
def remesh(
        vertices_etc:torch.Tensor, #V,D
        faces:torch.Tensor, #F,3 long
        face_mask,
        flip:bool,
        max_vertices = 1e6,
        threshold = 0.005,
        ):

    face_masked = faces[face_mask.squeeze(-1) == 1]
    unique_coords = torch.unique(face_masked) 
    vertex_mask = torch.zeros((vertices_etc.shape[0]), dtype = torch.bool).to(vertices_etc.device)
    vertex_mask[unique_coords.squeeze(-1)] = 1.0
    nan_tensor = torch.tensor([torch.nan],device=vertex_mask.device)
    vertex_mask = torch.concat((nan_tensor,vertex_mask))

    vertices_etc,faces = prepend_dummies(vertices_etc,faces)
    vertices = vertices_etc[:,:3] #V,3
    nan_tensor = torch.tensor([torch.nan],device=vertices_etc.device)

    def get_edge_length(v1_idx, v2_idx):
        return torch.norm(vertices_etc[v1_idx] - vertices_etc[v2_idx], dim = -1)


    edges,face_to_edge = calc_edges(faces) 
    vertices = vertices_etc[:,:3]
    edge_length = calc_edge_length(vertices,edges) #E
    splits = vertex_mask[edges].to(torch.float32).mean(dim=-1) == 1
    
    splits = torch.logical_and(splits,get_edge_length(edges[:,0],edges[:,1]) > threshold)
    vertices_etc,faces = split_edges(vertices_etc,faces,edges,face_to_edge,splits,pack_faces=False)

    vertices_etc,faces = pack(vertices_etc,faces)
    vertices = vertices_etc[:,:3]

    if flip:
        edges,_,edge_to_face = calc_edges(faces,with_edge_to_face=True) #E,2 F,3
        flip_edges(vertices,faces,edges,edge_to_face,with_border=False)

    vertices_etc,faces = remove_dummies(vertices_etc,faces)
    
    return vertices_etc,faces,face_mask