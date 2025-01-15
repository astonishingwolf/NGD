from matplotlib import pyplot as plt

# from script.losses.layers import NearestNeighbour
# from script.losses.utils import *
# from script.utils.io import get_garment_path, load_obj
from scipy.spatial.transform import Rotation as R
# from script.train.smpl import VertexNormals
# from script.utils.global_vars import ROOT_DIR

from scripts.utils.physics_utils   import *
from scripts.losses.physics import *
import trimesh 
import os
root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

thickness = 0.00047
bulk_density = 426.0
area_density = thickness * bulk_density
young_modulus=0.7e5
poisson_ratio = 0.485
stretch_multiplier = 1.0
bending_multiplier = 50.0

device = 'cuda'

class Material():
    '''
    This class stores parameters for the StVK material model
    '''

    def __init__(self, density,       # Fabric density (kg / m2)
                       thickness,     # Fabric thickness (m)
                       young_modulus, 
                       poisson_ratio,
                       bending_multiplier=1.0,
                       stretch_multiplier=1.0):
                       
        self.density = density
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

        self.bending_multiplier = bending_multiplier
        self.stretch_multiplier = stretch_multiplier

        # Bending and stretching coefficients (ARCSim)
        self.A = young_modulus / (1.0 - poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= stretch_multiplier

        self.bending_coeff = self.A / 12.0 * (thickness ** 3) 
        self.bending_coeff *= bending_multiplier

        # Lamé coefficients (μ=2.36e4 and λ=4.44e4)
        self.lame_mu = 0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio

        print(f"Lame mu {self.lame_mu:.2E}, Lame lambda: {self.lame_lambda:.2E}")
        

material_default = Material(
    density=area_density,  # Fabric density (kg / m2)
    thickness=thickness,  # Fabric thickness (m)
    young_modulus=young_modulus,
    poisson_ratio=poisson_ratio,
    stretch_multiplier=stretch_multiplier,
    bending_multiplier=bending_multiplier
)
        
class Cloth():
    """
    Stores mesh and material information of the garment
    """
    def __init__(self, garment_vertices, garment_faces, material = material_default, data_type=torch.float32):
        super(Cloth, self).__init__()
        
        self.data_type = data_type
        self.material = material

        # path = get_garment_path(self.args_expt['garment'])
        # path = os.path.join(ROOT_DIR, path)

        v = garment_vertices
        f = garment_faces
        # 
        self.normals = face_normals(v, f).cpu().numpy()
        
        # Vertex attributes
        self.v_template = v
        self.v_mass = get_vertex_mass(v.cpu(), f.cpu(), self.material.density, self.data_type)
        self.v_velocity = torch.zeros((1, v.shape[0], 3), dtype=self.data_type) 
        
        self.num_vertices = self.v_template.shape[0]
        self.v_weights = torch.zeros((self.num_vertices, 24), dtype=self.data_type)

        # Face attributes
        self.f = f
        self.f_connectivity = get_face_connectivity(f)           
        self.f_connectivity_edges = get_face_connectivity_edges(f) 
        self.f_area = torch.tensor(get_face_areas(v, f), dtype=self.data_type).to(device)
        self.num_faces = self.f.shape[0]

        # Edge attributes
        self.e = get_vertex_connectivity(f)                        
        self.e_rest = get_edge_length(v, self.e)            
        self.num_edges = self.e.shape[0]

        # print('num vertices:', self.num_vertices)
        # print('num faces:', self.num_faces)

        self.Dm_inv = self.make_continuum()
        self.Dm_inv = torch.tensor(self.Dm_inv, dtype=self.data_type).to(device)

        self.e_dict = get_v_connect_dict(self.e)
        # self.ragged_neigh = dict_to_ragged_tensor(self.e_dict)
        # self.isometry_covar = isometry_covar(torch.unsqueeze(self.v_template, dim=0), self.ragged_neigh)
        # self.eig_val, _ = torch.linalg.eigh(self.isometry_covar) 

    def compute_bending_coeff(self, smpl):
        va = self.v_template[None, :]
        vb = smpl.template_vertices[None, :]
        nb = VertexNormals()(vb, smpl.faces)
        closest_vertices = NearestNeighbour()(va, vb)

        vb = torch.gather(vb, 1, closest_vertices.unsqueeze(-1).expand(-1, -1, vb.shape[-1]))
        nb = torch.gather(nb, 1, closest_vertices.unsqueeze(-1).expand(-1, -1, nb.shape[-1]))

        distance = torch.sum(nb * (va - vb), dim=-1)

        dist_a = torch.gather(distance, 1, self.f_connectivity_edges[:, 0].unsqueeze(0))
        dist_b = torch.gather(distance, 1, self.f_connectivity_edges[:, 1].unsqueeze(0))
        dist_edge_body = 0.5 * (dist_a + dist_b)
        min_dist = torch.min(dist_edge_body)
        max_dist = torch.max(dist_edge_body)
        w_bend_template = (dist_edge_body - min_dist) / (max_dist - min_dist)
        return w_bend_template

    def make_continuum(self):
        """
        Calculate material space uv
        """
        f = self.f.cpu().numpy()
        angle = np.arccos(self.normals[:, 2])
        axis = np.stack(
            [
                self.normals[:, 1],
                -self.normals[:, 0],
                np.zeros((self.normals.shape[0],), np.float32),
            ],
            axis=-1,
        )
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        axis_angle = axis * angle[..., None]
        rotations = R.from_rotvec(axis_angle).as_matrix()
        triangles = self.v_template.cpu().numpy()[f]
        triangles = np.einsum("abc,adc->abd", triangles, rotations)

        triangles = triangles[..., :2]
        uv_matrices = np.stack(
            [triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 2]],
            axis=-1,
        )
        Dm_inv = np.linalg.inv(uv_matrices)
        return Dm_inv

    def map_indices_to_3D_space(self, em_dict, mapping):
        e_dict = {}
        seam_idx = []
        seam_idx_m = []
        mapping_inv = {}        # k to km mapping
        for k_m, v_m in em_dict.items():
            k = mapping[k_m]
            v = [mapping[vv] for vv in v_m]
            if k in e_dict:
                seam_idx_m.append(k_m)
                if k not in seam_idx:
                    seam_idx.append(k)
                if mapping_inv[k] not in seam_idx_m:
                    seam_idx_m.append(mapping_inv[k])
            else:
                e_dict[k] = v
                mapping_inv[k] = k_m
        for d in seam_idx:
            del e_dict[d]
        for d in seam_idx_m:
            del em_dict[d]
        return e_dict, em_dict

def face_normals(vertices, faces, normalized=True):
    
    # faces = faces.unsqueeze(0)
    
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    # 
    
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalized:
        face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
        
    return face_normals #F,3


if __name__ == "__main__":
    
    def load_garment(gament_model):
        mesh = trimesh.load(gament_model)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)
        return vertices, faces
    
    verts, faces = load_garment('/home/nakul/soham/3d/Garment3DGen/meshes/tshirt.obj')
    cloth = Cloth(verts,faces)
    stvk = StVKLoss(cloth)
    bend = BendingLoss(cloth)
    stvk_loss,_,_ = stvk(verts)
    bending_loss,_ = bend(verts)
    # 