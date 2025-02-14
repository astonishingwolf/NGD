import math
import numpy as np
import torch
from scripts.utils.physics_utils import *

device = 'cuda'

class EdgeLoss:
    def __init__(self, weight, template):
        
        self.weight = weight
        self.el_template = template.e_rest
        self.e = template.e
        self.f_area = template.f_area
        self.f = template.f

    def __call__(self, v):
        
        batch_size = v.shape[0]

        el_garment = get_edge_length(v, self.e)
        edge_difference = el_garment - self.el_template

        loss = torch.sum(edge_difference ** 2) / batch_size

        diff_el_pct = torch.abs(edge_difference) / self.el_template
        metric = torch.mean(diff_el_pct)  # edge elongation in %

        area_garment = get_face_areas_batch(v, self.f)
        diff_area = torch.abs(area_garment - self.f_area)
        diff_area_pct = diff_area / self.f_area
        metric_area = torch.mean(diff_area_pct)

        return loss, metric, metric_area

# Saint-Venant Kirchhoff
class StVKLoss:
    def __init__(self, weight, params):
        
        self.weight = weight
        template = params.cloth
        self.template = template

    def __call__(self, v):
        
        batch_size = v.shape[0]
        # 
        triangles = gather_triangles(v, self.template.f)
        # Dm_inv = self.template.Dm_inv.unsqueeze(0).repeat(v.shape[0], 1, 1, 1)
        Dm_inv = self.template.Dm_inv
        F = deformation_gradient(triangles, Dm_inv)
        G = green_strain_tensor(F)

        # Energy
        mat = self.template.material
        I = torch.eye(2, dtype=G.dtype, device=G.device).unsqueeze(0).repeat(G.shape[0], 1, 1)
        G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1) 
        # S = mat.lame_mu * G + 0.5 * mat.lame_lambda * torch.trace(G, dim1=-2, dim2=-1).unsqueeze(-1).unsqueeze(-1) * I
        # energy_density = torch.trace(S.transpose(-1, -2) @ G, dim1=-2, dim2=-1)
        # 
        S = mat.lame_mu * G + 0.5 * mat.lame_lambda * G_trace[:, None, None] * I
        energy_density_matrix = S.permute(0, 2, 1) @ G
        energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        energy = self.template.f_area.unsqueeze(0) * mat.thickness * energy_density
        loss = torch.sum(energy) / batch_size

        el_template = self.template.e_rest
        el_garment = get_edge_length(v, self.template.e)
        diff_el = torch.abs(el_garment - el_template)
        diff_el_pct = diff_el / el_template
        metric_edge = torch.mean(diff_el_pct)       # edge difference in %
        # 
        area_garment = get_face_areas_batch(v, self.template.f)
        diff_area = torch.abs(area_garment - self.template.f_area)
        diff_area_pct = diff_area / self.template.f_area
        metric_area = torch.mean(diff_area_pct)     # area difference in %

        return loss, metric_edge, metric_area

class BendingLoss:

    def __init__(self, cfg, weight, params, follow_template_weight=None):

        self.weight = weight
        template = params.cloth
        self.template = template
        v_template = torch.tensor(self.template.v_template[np.newaxis, :], dtype=torch.float32).to(device)

        fn = FaceNormals()(v_template, self.template.f).unsqueeze(0)
        
        # 
        self.template.f_connectivity =  self.template.f_connectivity.to(torch.int64)
        n0 = torch.index_select(fn, 1, self.template.f_connectivity[:, 0])
        n1 = torch.index_select(fn, 1, self.template.f_connectivity[:, 1])
        v0 = torch.index_select(v_template, 1, self.template.f_connectivity_edges[:, 0])
        v1 = torch.index_select(v_template, 1, self.template.f_connectivity_edges[:, 1])
        
        e = v1 - v0
        e_norm = torch.nn.functional.normalize(e, dim=-1)
        cos = torch.sum(n0 * n1, dim=-1)
        sin = torch.sum(e_norm * torch.cross(n0, n1, dim=-1), dim=-1)
        self.theta_template = torch.atan2(sin, cos)

        self.follow_template_weight = follow_template_weight

        
    def __call__(self, v):
        batch_size = v.shape[0]

        fn = FaceNormals()(v, self.template.f)
        fn = fn.unsqueeze(0)
        v = v.unsqueeze(0)
        self.template.f_connectivity = self.template.f_connectivity.to(torch.int64)
        self.template.f_connectivity_edges = self.template.f_connectivity_edges.to(torch.int64)
        # 
        n0 = torch.gather(fn, 1, self.template.f_connectivity[:, 0].unsqueeze(-1).repeat(1, 1, fn.shape[-1]))
        n1 = torch.gather(fn, 1, self.template.f_connectivity[:, 1].unsqueeze(-1).repeat(1, 1, fn.shape[-1]))

        v0 = torch.gather(v, 1, self.template.f_connectivity_edges[:, 0].unsqueeze(-1).repeat(1, 1, v.shape[-1]))
        v1 = torch.gather(v, 1, self.template.f_connectivity_edges[:, 1].unsqueeze(-1).repeat(1, 1, v.shape[-1]))
        e = v1 - v0
        e_norm = torch.nn.functional.normalize(e, dim=-1)
        l = torch.norm(e, dim=-1, keepdim=True)

        f_area = self.template.f_area.unsqueeze(0).repeat(v.shape[0], 1)
        # 
        a0 = torch.gather(f_area, 1, self.template.f_connectivity[:, 0].unsqueeze(0))
        a1 = torch.gather(f_area, 1, self.template.f_connectivity[:, 1].unsqueeze(0))
        a = a0 + a1

        cos = torch.sum(n0 * n1, dim=-1)
        sin = torch.sum(e_norm * torch.cross(n0, n1, dim=-1), dim=-1)
        theta = torch.atan2(sin, cos)

        mat = self.template.material
        scale = l[..., 0] ** 2 / (4 * a)

        energy1 = mat.bending_coeff * scale * (theta ** 2) / 2
        energy2 = mat.bending_coeff * scale * ((self.theta_template - theta) ** 2) / 2
        if self.follow_template_weight is not None:
            energy = (1 - self.follow_template_weight) * energy1 + self.follow_template_weight * energy2
        else:
            energy = 0.5 * energy1 + 0.5 * energy2

        loss = torch.sum(energy) / batch_size
        metric = torch.mean(torch.abs(theta)) * 180 / math.pi
        return loss, metric