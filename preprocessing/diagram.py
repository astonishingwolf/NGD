import torch
import trimesh
import matplotlib.pyplot as plt

def tensor_to_colormap(tensor):
    """
    Normalize a 1D PyTorch tensor and map it to the Jet colormap.
    """
    tensor_min, tensor_max = tensor.min(), tensor.max()
    tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)  # Avoid division by zero
    tensor_np = tensor_norm.cpu().numpy()
    cmap = plt.get_cmap("jet")
    colors = cmap(tensor_np)[:, :3]  # Extract RGB channels
    return torch.tensor(colors * 255, dtype=torch.uint8)

# Load tensor containing scalar values for faces
tensor = torch.load('/hdd_data/nakul/soham/New_Results/Dress4D/Demo/Remeshed/upper/Final_Diagram/RemeshingDiagram/400.pt')

# Convert tensor to RGB colors
color_tensor = tensor_to_colormap(tensor.squeeze(-1))

# Load mesh
mesh = trimesh.load('/hdd_data/nakul/soham/New_Results/Dress4D/Demo/Remeshed/upper/Final_Diagram/remeshed_template/original_mesh.obj')  # Change path to your mesh file

# Ensure color tensor matches the number of faces
if color_tensor.shape[0] != len(mesh.faces):
    raise ValueError("Number of colors does not match number of faces")

# Apply colors to the mesh
mesh.visual.face_colors = color_tensor.numpy()

# Save the mesh with face colors
mesh.export('/hdd_data/nakul/soham/New_Results/Dress4D/Demo/Remeshed/upper/Final_Diagram/RemeshingDiagram/colored_mesh.obj')  # Change path to desired save location
