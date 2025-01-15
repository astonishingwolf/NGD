import pymeshlab
import torch
import os
from packages.nvdiffmodeling.src import obj
from packages.nvdiffmodeling.src import mesh
from packages.nvdiffmodeling.src import texture
import numpy as np
import igl
import torch 
import torchvision.transforms as transforms
import numpy as np
import os
import trimesh
from PIL import Image
# import util as util
import nvdiffrast.torch as dr
import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from natsort import natsorted
import re
from matplotlib import pyplot as plt
import torchvision.transforms.functional as vis_F
import torch.nn.functional as F
# import pygeodesic.geodesic as geodesic
import cv2

def get_vp_map(v_pos, mtx_in, resolution):
    device = v_pos.device
    with torch.no_grad():
        vp_mtx = torch.tensor([
            [resolution / 2, 0., 0., (resolution - 1) / 2],
            [0., resolution / 2, 0., (resolution - 1) / 2],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.,]
        ], device=device)

        v_pos_clip = ru.xfm_points(v_pos[None, ...], mtx_in)
        v_pos_div = v_pos_clip / v_pos_clip[..., -1:]

        v_vp = (vp_mtx @ v_pos_div.transpose(1, 2)).transpose(1, 2)[..., :-1]

        # don't need manual z-buffer here since we're using the rast map to do occlusion
        if False:
            v_pix = v_vp[..., :-1].int().cpu().numpy()
            v_depth = v_vp[..., -1].cpu().numpy()

            # pix_v_map = -torch.ones(len(v_pix), resolution, resolution, dtype=int)
            pix_v_map = -np.ones((len(v_pix), resolution, resolution), dtype=int)
            # v_pix_map = resolution * torch.ones(len(v_pix), len(v_pos), 2, dtype=int)
            v_pix_map = resolution * np.ones_like(v_pix, dtype=int)
            # buffer = torch.ones_like(pix_v_map) / 0
            buffer = -np.ones_like(pix_v_map) / 0
            for i, vs in enumerate(v_pix):
                for j, (y, x) in enumerate(vs):
                    if x < 0 or x > resolution - 1 or y < 0 or y > resolution - 1:
                        continue
                    else:
                        if v_depth[i, j] > buffer[i, x, y]:
                            buffer[i, x, y] = v_depth[i, j]
                            if pix_v_map[i, x, y] != -1:
                                v_pix_map[i, pix_v_map[i, x, y]] = np.array([resolution, resolution])
                            pix_v_map[i, x, y] = j
                            v_pix_map[i, j] = np.array([x, y])
            v_pix_map = torch.tensor(v_pix_map, device=device)
        v_pix_map = v_vp[..., :-1].int().flip([-1])
        v_pix_map [(v_pix_map > resolution - 1) | (v_pix_map < 0)] = resolution
    return v_pix_map.long()

texture_map = texture.create_trainable(np.random.uniform(size=[512] * 2 + [3], low=0.0, high=1.0), [512] * 2, True)
normal_map = texture.create_trainable(np.array([0, 0, 1]), [512] * 2, True)
specular_map = texture.create_trainable(np.array([0, 0, 0]), [512] * 2, True)

def get_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj'):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        # some arbitrarily high number
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    if not os.path.exists(os.path.join(output_path, 'tmp')):
        os.makedirs(os.path.join(output_path, 'tmp'))

    ms.save_current_mesh(os.path.join(output_path, 'tmp', mesh_name))
    
    load_mesh = obj.load_obj(os.path.join(output_path, 'tmp', mesh_name))
    load_mesh = mesh.unit_size(load_mesh)

    ms.add_mesh(
        pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(os.path.join(output_path, 'tmp', mesh_name), save_vertex_color=False)

    load_mesh = mesh.Mesh(
        material={
            'bsdf': bsdf_flag,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh  # Get UVs from original loaded mesh
    )
    return load_mesh


def get_og_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj'):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        # some arbitrarily high number
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    ms.save_current_mesh(str(output_path / 'tmp' / mesh_name))

    load_mesh = obj.load_obj(str(output_path / 'tmp' / mesh_name))
    load_mesh = mesh.resize_mesh(load_mesh)

    ms.add_mesh(
        pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / mesh_name), save_vertex_color=False)

    load_mesh = mesh.Mesh(
        material={
            'bsdf': bsdf_flag,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh  # Get UVs from original loaded mesh
    )
    return load_mesh

def compute_mv_cl(final_mesh, fe, normalized_clip_render, params_camera, train_rast_map, cfg, device):
    # Consistency loss
    # Get mapping from vertex to pixels
    curr_vp_map = get_vp_map(final_mesh.v_pos, params_camera['mvp'], 224)
    for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
        u_faces = rast_faces.unique().long()[1:] - 1
        t = torch.arange(len(final_mesh.v_pos), device=device)
        u_ret = torch.cat([t, final_mesh.t_pos_idx[u_faces].flatten()]).unique(return_counts=True)
        non_verts = u_ret[0][u_ret[1] < 2]
        curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)

    # Get mapping from vertex to patch
    med = (fe.old_stride - 1) / 2
    curr_vp_map[curr_vp_map < med] = med
    curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
    curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
    flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]

    # Deep features
    patch_feats = fe(normalized_clip_render)
    flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
    flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

    deep_feats = patch_feats[cfg.consistency_vit_layer]
    deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
    deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
    deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

    elev_d = torch.cdist(params_camera['elev'].unsqueeze(1), params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_elev_filter))
    azim_d = torch.cdist(params_camera['azim'].unsqueeze(1), params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_azim_filter))

    cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
    cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
    consistency_loss = cosines[cosines != 0].mean()
    return consistency_loss

def load_images(images_path, type = 'image', resolution = (512,512), device = 'cuda'):

        image_list = []
        trans = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor()
        ])
        image_filenames = natsorted(os.listdir(images_path))
        # print(image_filenames)
        for image in image_filenames:
            image_path = os.path.join(images_path, image)
            if type == 'image':
                image = Image.open(image_path).convert('RGB')
            elif type == 'mask':
                image = Image.open(image_path).convert('L')
            # image = Image.open(image_path).convert('RGB')
            image = trans(image)
            image_list.append(image)
        image_tensor = torch.stack(image_list)
        image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.permute(0,2,3,1)
        image_tensor = image_tensor.to(torch.float32)
        
        return image_tensor
    
def similarity_descriptors(similarity_path, DEVICE = 'cuda'):
    pass



def modify_embeddded_graph(verts,faces,Nodes,weights,node_displacement):
    
    displacements = torch.matmul(weights, node_displacement)
    new_vertices = verts + displacements
    
    return new_vertices , faces

def save_mesh(mesh_verts,mesh_faces, mesh_path):

    mesh_verts = mesh_verts.detach()
    mesh_faces = mesh_faces.detach()
    mesh_verts = mesh_verts.cpu().numpy()
    mesh_faces = mesh_faces.cpu().numpy()
    mesh = trimesh.Trimesh(vertices = mesh_verts, faces = mesh_faces)
    
    mesh.export(mesh_path)
    print(f"Mesh saved to {mesh_path}")

def save_image_epoch(image_batch, type = 'textured_image', save_image_path = None, start_iter=0, batch_idx = None):

    os.makedirs(save_image_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_image_path, type)):
        os.makedirs(os.path.join(save_image_path, type))   
    
    save_image_path = os.path.join(save_image_path, type)      
    to_pil = transforms.ToPILImage()
    if image_batch.dim() == 4 or type in ['mask','mask_inter']:
        for i, image_tensor in enumerate(image_batch):
            iter = batch_idx[i]
            # print(f"Iteration saved  {iter}")
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
                
            image = to_pil(image_tensor.cpu())
            image.save(os.path.join(save_image_path, f'infer{iter:04d}.jpg'))
    else :
        if image_batch.dim() == 3:
            image_tensor = image_batch.permute(2, 0, 1)
            image = to_pil(image_tensor.cpu())
            image.save(os.path.join(save_image_path, f'infer{start_iter:04d}.jpg'))

def load_mesh_path(mesh_path, scale = 0.8, device = 'cuda'):
    """
    Loading a mesh from a a path
    TODO : Extract the texture if given 
    """
    if mesh_path.endswith(".obj"):
        # mesh = trimesh.load_mesh(mesh_path, 'obj')
        # V, F = mesh.vertices, mesh.faces
        V, _, _, F, _, _ =igl. read_obj(mesh_path)
    elif mesh_path.endswith(".stl"):
        mesh = trimesh.load_mesh(mesh_path, 'stl')    
    elif mesh_path.endswith(".ply"):
        mesh = trimesh.load_mesh(mesh_path, 'ply')
    else:
        raise ValueError(f"unknown mesh file type: {mesh_path}")

    verts = torch.tensor(V, dtype = torch.float32, device = device)
    faces = torch.tensor(F, dtype = torch.int64 , device = device)
    # breakpoint()
    # colors = torch.tensor(mesh.visual.vertex_colors, dtype = torch.float32, device = device)
    # if scale > 0:
    #     print(f"The scale is working {scale}")
    #     target_vertices = verts - verts.mean(dim=0, keepdim=True)
    #     max_norm = torch.max(torch.norm(target_vertices, dim=-1)) + 1e-6
    #     verts = (target_vertices / max_norm) * scale

    print(f"The vertices shape {verts.shape}")
    return verts,faces

def load_data(mesh_path, scale = 0.8,device = 'cuda'):
    """
    Loading the meshes and images from the dataset
    """
    # images_tensor = load_images(images_path,device = device)
    mesh_trimesh, verts, faces = load_mesh(mesh_path,scale = scale, device = device )
    mesh = {
        "vertices" : verts,
        "faces" : faces
    }
    return mesh  , mesh_trimesh

def ZEROBS_OBJ_READER(file_path):
    
    f = open(file_path)

    v = []
    vt = []
    vn = []
    f_v = []
    f_vt = []
    f_vn = []

    for line in f:
        line = re.sub(' +', ' ',line) #removes unwanted spaces from string
        if line[0:2]=='v ':
            info = line.strip().split(" ")
            v.append([float(info[1]),float(info[2]),float(info[3])])
        
        elif line[0:2]=='vt':
            info = line.strip().split(" ")
            vt.append([float(info[1]),float(info[2])])
        
            
        elif line[0:2]=='vn':
            info = line.strip().split(" ")
            vn.append([float(info[1]),float(info[2]),float(info[3])])

        elif line[0:2]=='f ':
            info = line.strip().split(" ")
            face1 = info[1].split("/")
            face2 = info[2].split("/")
            face3 = info[3].split("/")
            verts_idx = [int(face1[0]) -1,int(face2[0]) -1,int(face3[0]) -1]
            f_v.append(verts_idx)
            if min(len(face1),len(face2),len(face3))==3:
                verts_tex_idx = [int(face1[1]) -1,int(face2[1]) -1,int(face3[1]) -1]
                verts_norm_idx = [int(face1[2]) -1,int(face2[2]) -1,int(face3[2]) -1]
                f_vt.append(verts_tex_idx)
                f_vn.append(verts_norm_idx)

    f.close()
    v = np.array(v)
    vt = np.array(vt)
    vn = np.array(vn)
    f_v = np.array(f_v)
    f_vt = np.array(f_vt)
    f_vn = np.array(f_vn)

    mesh_info = {'v':v,'vt':vt,'vn':vn,'f_v':f_v,'f_vt':f_vt,'f_vn':f_vn}
    return mesh_info

def dowmsample_mesh(mesh, target_reduction = 0.05, scale = 0.8, device = 'cuda'):

    target_faces = int(len(mesh.faces) * target_reduction)
    simplified_mesh = mesh.simplify_quadratic_decimation(target_faces)
    verts = torch.tensor(simplified_mesh.vertices, dtype = torch.float32, device = device)
    faces = torch.tensor(simplified_mesh.faces , dtype = torch.int32 , device = device)

    if scale > 0:
        print(f"The scale is working {scale}")
        target_vertices = verts - verts.mean(dim=0, keepdim=True)
        max_norm = torch.max(torch.norm(target_vertices, dim=-1)) + 1e-6
        verts = (target_vertices / max_norm) * scale

    print(f"The vertices shape {verts.shape}")
    mesh = {
        "vertices" : verts,
        "faces" : faces
    }
    
    return mesh

def triangles_to_edges(faces: torch.Tensor):
    """Computes mesh edges from triangles."""

    edges_list = [faces[..., 0:2],
                  faces[..., 1:3],
                  torch.stack([faces[..., 2], faces[..., 0]], dim=-1)]
    edges = torch.cat(edges_list, dim=1)
    receivers = edges.min(dim=-1)[0]
    senders = edges.max(dim=-1)[0]

    packed_edges = torch.stack([senders, receivers], dim=-1)
    unique_edges = torch.unique(packed_edges, dim=1)

    sortvals = unique_edges[..., 0] * 10000 + unique_edges[..., 1]
    sort_idx = torch.sort(sortvals, dim=1).indices[0]
    unique_edges = unique_edges[:, sort_idx]
    senders, receivers = torch.unbind(unique_edges, dim=-1)

    # create two-way connectivity
    all_senders = torch.cat([senders, receivers], dim=1)
    all_receivers = torch.cat([receivers, senders], dim=1)

    edges = torch.cat([all_senders, all_receivers], dim=0)
    return edges

def get_texture(img, resolution=512, device = 'cuda'):
    # Image.open(img).convert('RGB').resize((resolution, resolution))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p = 1),
    ])
    image_tensor = transform(Image.open(img).convert('RGB').resize((resolution, resolution)))
    image_tensor = image_tensor.to(torch.float32).to(device)
    image_tensor = image_tensor.permute(1,2,0)
    return image_tensor

def load_textures(texture_path, device = 'cuda'):
    mesh_info = ZEROBS_OBJ_READER(texture_path)
    vert_tex = torch.from_numpy(mesh_info['vt']).to(device).to(torch.float32)
    face_tex = torch.from_numpy(mesh_info['f_vt']).to(device).to(torch.int32)
    texture_path  = texture_path.split('.')[0] + '.png'
    tex = get_texture(texture_path)

    return vert_tex, face_tex, tex

def extract_dino_features(images_batch, load_size : int = 224, layer : int = 11, facet :str = 'key', \
    bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',device = 'cuda'):
    extractor = ViTExtractor(model_type, stride, device=device)
    images = extractor.preprocess_tensors(images_batch, load_size)
    descs_feat = extractor.extract_descriptors(images.to(device), layer, facet, bin, include_cls=True)
    return descs_feat

# def extract_dino_features_inChunks(images_batch, chunks = 4, load_size : int = 224, layer : int = 11, facet :str = 'key', \
#     bin: bool = False, stride: int = 4, model_type: str = 'dino_vits8',device = 'cuda'):
    
#     for i, image_chunk in enumerate(images_batch,4)
#     extractor = ViTExtractor(model_type, stride, device=device)
#     images = extractor.preprocess_tensors(images_batch, load_size)
#     descs_feat = extractor.extract_descriptors_inChunks(images.to(device), layer, facet, bin, include_cls=True)
#     return descs_feat

def mask_interxsection(mask_pred, mask_gt):
    
    dims = tuple((range(mask_pred.ndimension())[1:]))
    intersect = torch.logical_and(mask_pred, mask_gt)
    print(intersect.shape)
    union = (mask_pred + mask_gt - mask_pred * mask_gt).sum(dims) + 1e-6
    print(union.shape)
    return 1.0 - (intersect / union).sum(dims) / intersect.nelement()

class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
    
def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[2:]
    flow_resized = cv2.resize(flow[0].permute(1, 2, 0).cpu().detach().numpy(), (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= neww / oldw
    flow_resized[:, :, 1] *= newh / oldh
    flow_resized = torch.from_numpy(flow_resized).permute(2, 0, 1).to(flow.device).unsqueeze(0)
    return flow_resized

def plot(imgs, iterator, save_dir,**imshow_kwargs):
    
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    for i,img in enumerate(imgs):
        
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=1, ncols=num_cols, squeeze=False)
        for col_idx, img in enumerate(img):
            ax = axs[0, col_idx]
            img = vis_F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{iterator[i]}.png'))
    plt.close('all')
    
def preprocess(img1_batch, img2_batch, device = 'cuda'):
    
    if img1_batch.shape[-3] != 3:
        img1_batch = img1_batch.permute(0,3,1,2)
        img2_batch = img2_batch.permute(0,3,1,2)
        
    img1_batch = vis_F.resize(img1_batch, size=[512, 512], antialias=False) 
    img2_batch = vis_F.resize(img2_batch, size=[512, 512], antialias=False) 
    
    img1_batch = img1_batch.contiguous()
    img2_batch = img2_batch.contiguous()
    
    if img1_batch.is_contiguous is False or img2_batch.is_contiguous is False:
        # print(f"Changing the input tensors")
        img1_batch = img1_batch.contiguous_()
        img2_batch = img2_batch.contiguous_()
        
        # print(img1_batch.shape)
        # print(f"Input tensors are already contiguous")
        
    if img1_batch.max() > 1:
        
        img1_batch = img1_batch / 255  
        img2_batch = img2_batch / 255
    
    img1_batch = img1_batch.to(torch.float32).to(device)
    img2_batch = img2_batch.to(torch.float32).to(device)
    return img1_batch, img2_batch
    
def save_optical_flow(predicted_flow,texture_map_gt,texture_map_pred,iterator,save_image_path):
    
    type = 'flow'
    if not os.path.exists(os.path.join(save_image_path, type)):
        os.makedirs(os.path.join(save_image_path, type))   
        
    save_image_path = os.path.join(save_image_path, type)
    
    flow_imgs = flow_to_image(predicted_flow)
    img1_batch, img2_batch = preprocess(texture_map_gt, texture_map_pred)    
    img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
    img2_batch = [(img2 + 1) / 2 for img2 in img2_batch]
    grid = [[img1, img2, flow_img] for (img1, img2, flow_img) in zip(img1_batch, img2_batch, flow_imgs)]

    plot(grid,iterator,save_image_path)
    
    
if __name__=='__main__':
    
    # random_images = torch.rand(64,3,512,512)
    # num_chunks = 16
    # images_inChunks = torch.chunk(random_images, num_chunks, dim = 0)
    # feat = []
    # for images_batch in images_inChunks:
    #     with torch.no_grad():
    #         feat.append(extract_dino_features(images_batch))
    # feat = torch.stack(feat)
    # feat = feat.view(feat.shape[0]*feat.shape[1],feat.shape[2],feat.shape[3],feat.shape[4])
    # print(feat.shape)
    shape = (5,64,64)
    
    mask_pred = torch.randint(0, 2, shape).float()
    mask_gt = torch.randint(0, 2, shape).float()
    print(mask_gt)
    print(mask_pred)
    print(mask_interxsection(mask_pred, mask_gt))
    
    
def change_scaling(target_vertices, scale=0.8):
    
    if scale > 0:
        target_vertices = target_vertices - target_vertices.mean(dim=0, keepdim=True)
        max_norm = th.max(th.norm(target_vertices, dim=-1)) + 1e-6
        target_vertices = (target_vertices / max_norm) * scale
    # print(f"The target mesh {target_vertices.shape}")
    return target_vertices

def change_scaling_np(target_vertices, scale=0.8):
    
    if scale > 0:
        target_vertices = target_vertices - np.mean(target_vertices,axis=0, keepdims=True)
        max_norm = np.max(np.linalg.norm(target_vertices, axis=-1)) + 1e-6
        target_vertices = (target_vertices / max_norm) * scale
    # print(f"The target mesh {target_vertices.shape}")
    return target_vertices

def calculate_face_normals(mesh_vertices, mesh_faces):
     
    v0 = mesh_vertices[mesh_faces[:, 0]]
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]

    # Calculate face normals
    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Normalize face normals
    face_normals = F.normalize(face_normals, p=2, dim=1)

    return face_normals

def calculate_face_centers(mesh_vertices, mesh_faces):
    # Extract vertices for each face
    v0 = mesh_vertices[mesh_faces[:, 0]]
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]

    # Calculate face centers
    face_centers = (v0 + v1 + v2) / 3.0

    return face_centers
    
def merge_close_vertices(vertices, faces, tolerance=1e-6):
    """
    Merge close vertices and remove degenerate triangles from a mesh.
    
    Args:
        vertices: Tensor of shape (V, 3) containing vertex coordinates
        faces: Tensor of shape (F, 3) containing face indices
        tolerance: Distance threshold for merging vertices
    
    Returns:
        unique_vertices: Tensor of unique vertices
        filtered_faces: Tensor of valid faces with updated indices
    """
    # Ensure faces is 2D
    if faces.dim() == 3:
        faces = faces.squeeze(0)
    
    # Round vertices to merge close ones
    rounded_vertices = torch.round(vertices / tolerance) * tolerance
    
    # Find unique vertices and get mapping
    unique_vertices, inverse_indices = torch.unique(rounded_vertices, dim=0, return_inverse=True)
    
    # Update face indices
    updated_faces = inverse_indices[faces]
    
    # Create mask for valid (non-degenerate) faces
    valid_faces_mask = (updated_faces[:, 0] != updated_faces[:, 1]) & \
                       (updated_faces[:, 1] != updated_faces[:, 2]) & \
                       (updated_faces[:, 0] != updated_faces[:, 2])
    
    # Filter out degenerate faces
    filtered_faces = updated_faces[valid_faces_mask]
    
    # If no valid faces remain, return empty tensors or handle as needed
    if filtered_faces.shape[0] == 0:
        return unique_vertices, torch.zeros((0, 3), dtype=faces.dtype, device=faces.device)
    
    # Find actually used vertices
    used_vertex_mask = torch.zeros(unique_vertices.shape[0], dtype=torch.bool, device=vertices.device)
    used_vertex_mask[filtered_faces.flatten()] = True
    
    final_vertices = unique_vertices[used_vertex_mask]
    vertex_index_map = torch.zeros(unique_vertices.shape[0], dtype=torch.long, device=vertices.device)
    vertex_index_map[used_vertex_mask] = torch.arange(used_vertex_mask.sum(), device=vertices.device)
    final_faces = vertex_index_map[filtered_faces]
    # breakpoint()
    return final_vertices, final_faces

def jacobian_interpolation(jacobians_orig, orig_vertices, changed_vertices, \
    orig_faces, changed_faces, k = 3):

    face_centers_orig = calculate_face_centers(orig_vertices, orig_faces)
    face_centers_changed = calculate_face_centers(changed_vertices, changed_faces)
    
    distances = torch.cdist(face_centers_orig, face_centers_changed)
    k_distances, k_indices = distances.topk(k, dim = 1, largest = False)

    eps = 1e-10
    # breakpoint()
    weights = 1 / (k_distances + eps)
    weights = weights / torch.sum(weights, dim = 1, keepdim = True)

    weights = weights[None, ..., None , None]

    nearest_features = jacobians_orig[:,k_indices]
    # breakpoint()
    interpolated_features = torch.sum(nearest_features*weights, dim = 2)

    return interpolated_features

def optimizer_interpolation(jacobians_orig, orig_vertices, changed_vertices, \
    orig_faces, changed_faces, k = 3):

    face_centers_orig = calculate_face_centers(orig_vertices, orig_faces)
    face_centers_changed = calculate_face_centers(changed_vertices, changed_faces)
    
    distances = torch.cdist(face_centers_orig, face_centers_changed)
    k_distances, k_indices = distances.topk(k, dim = 1, largest = False)

    eps = 1e-10
    # breakpoint()
    weights = 1 / (k_distances + eps)
    weights = weights / torch.sum(weights, dim = 1, keepdim = True)

    weights = weights[None, ..., None , None]

    nearest_features = jacobians_orig[:,k_indices]
    # breakpoint()
    interpolated_features = torch.sum(nearest_features*weights, dim = 2)

    return interpolated_features


def merge_close_vertices_only(vertices, faces, tolerance=1e-6):
    """
    Merge close vertices in a mesh without removing degenerate faces.
    
    Args:
        vertices: Tensor of shape (V, 3) containing vertex coordinates
        faces: Tensor of shape (F, 3) containing face indices
        tolerance: Distance threshold for merging vertices
    
    Returns:
        unique_vertices: Tensor of unique vertices
        updated_faces: Tensor of faces with updated indices
    """
    # Ensure faces is 2D
    if faces.dim() == 3:
        faces = faces.squeeze(0)
    
    # Round vertices to merge close ones
    rounded_vertices = torch.round(vertices / tolerance) * tolerance
    
    # Find unique vertices and get mapping
    unique_vertices, inverse_indices = torch.unique(rounded_vertices, dim=0, return_inverse=True)
    
    # Update face indices
    updated_faces = inverse_indices[faces]
    # breakpoint()
    return unique_vertices, updated_faces


def mesh_surface_sampling_and_center(vertices, faces, device = 'cuda'):

    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    num = vertices.shape[0] / 100
    num = int(num)
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    sampled_points,_ = trimesh.sample.sample_surface_even(mesh,num)
    # breakpoint()
    mean = np.mean(sampled_points, axis=0)
    mean_tensor = torch.from_numpy(mean).float().to(device)
    
    return mean_tensor