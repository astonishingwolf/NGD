import numpy as np
import os
from PIL import Image

def main():
    raw_dir = '/data/guest/src/DGarment/Data/Dress4D/arnan_dance/sapiens_seg/sapiens_1b' 
    output_dir = '/data/guest/src/DGarment/Data/Dress4D/arnan_dance/shils'  
    os.makedirs(output_dir, exist_ok=True)

    seg_files = [file for file in os.listdir(raw_dir) if (file.endswith('.npy') & (file.split('.')[0].split('_')[-1] == 'seg'))]
    # breakpoint()
    for seg_file in seg_files:
        seg_file_npy = np.load(os.path.join(raw_dir, seg_file))

        seg_file_npy[(seg_file_npy != 22) & (seg_file_npy != 12)] = 0
        seg_file_npy[seg_file_npy == 22] = 1
        seg_file_npy[seg_file_npy == 12] = 1

        seg_file_npy = (seg_file_npy * 255).astype(np.uint8)

        img = Image.fromarray(seg_file_npy)

        output_path = os.path.join(output_dir, seg_file.replace('.npy', '.png'))
        img.save(output_path)

 
        print(f"Processed and saved: {output_path}")

def combine_segmentation_maps(png_dir, npy_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    png_files = [file for file in os.listdir(png_dir) if file.endswith('.png')]

    for png_file in png_files:
        # npy_file = png_file.split('_')[0] + '.npy'
        npy_file = png_file.replace("_seg.png", "") + ".npy"
        npy_file = npy_file.replace("anran_dance_", "anran_dance_f")
        npy_path = os.path.join(npy_dir, npy_file)
        # breakpoint()
        if not os.path.exists(npy_path):
            print(f"Warning: Corresponding NPY file not found for {png_file}. Skipping.")
            continue

        png_path = os.path.join(png_dir, png_file)
        png_seg = np.array(Image.open(png_path))  
        npy_seg = np.load(npy_path)              

        combined_seg = np.logical_or(png_seg, np.expand_dims(npy_seg, axis=-1)).astype(np.uint8) * 255


        combined_img = Image.fromarray(combined_seg)
        combined_img.save(os.path.join(output_dir, png_file))

        print(f"Processed and saved: {png_file}")

def apply_mask_to_normals(input_normals_dir, input_segmentation_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    normal_files = [f for f in os.listdir(input_normals_dir) if f.endswith('.png')]
    for normal_file in normal_files:
        normal_path = os.path.join(input_normals_dir, normal_file)
        normals = np.load(normal_path)
        normals = (normals + 1)/2
        seg_path_basename = normal_file.split('.')[0] + '.png'
        seg_path = os.path.join(input_segmentation_dir, seg_path_basename)
        if not os.path.exists(seg_path):
            print(f"Segmentation file missing for {normal_file}, skipping...")
            continue
        seg_mask = Image.open(seg_path).convert("L")  # Convert to grayscale
        seg_mask = np.array(seg_mask) > 0  # Binary mask
        masked_normals = (normals * seg_mask[..., np.newaxis]*255).astype(np.uint8) # Add channel axis
        output_path = os.path.join(output_dir, f"{normal_file.split('.')[0]}.png")
        Image.fromarray(masked_normals).save(output_path)
        print(f"Saved masked normal: {output_path}")

def fix_diffuse_shil(shil_dir, diffuse_dir, output_dir):
    os.makedirs(diffuse_dir, exist_ok=True)
    shil_files = [f for f in os.listdir(shil_dir) if f.endswith('.png')]
    for shil_file in shil_files:
        shil_path = os.path.join(shil_dir, shil_file)
        shil_image = Image.open(shil_path).convert("L")
        shil_image = np.array(shil_image) > 0 
        diffuse_file = os.path.join(diffuse_dir, f"{shil_file.split('.')[0].split('_')[0]}.png")
        diffuse_image = Image.open(diffuse_file).convert("L")
        diffuse_image = np.array(diffuse_image)
        masked_diffuse = (diffuse_image * shil_image).astype(np.uint8) 
        output_path = os.path.join(output_dir, f"{shil_file.split('.')[0].split('_')[0]}.png")
        masked_diffuse = masked_diffuse[..., np.newaxis]
        # breakpoint()
        Image.fromarray(masked_diffuse[..., [-1,-1,-1]]).save(output_path)
        print(f"Saved masked normal: {output_path}")

def fix_upper_shil(shil_lower_dir, shil_upper_dir):
    os.makedirs(shil_upper_dir, exist_ok=True)
    shil_lower_files = [f for f in os.listdir(shil_lower_dir) if f.endswith('.png')]
    for shil_lower_file in shil_lower_files:
        # Paths for lower silhouette image and corresponding upper silhouette numpy file
        shil_lower_path = os.path.join(shil_lower_dir, shil_lower_file)
        shil_upper_file = os.path.join(shil_upper_dir, shil_lower_file.replace(".png", ".npy"))

        # Load lower silhouette as binary mask
        shil_lower_image = Image.open(shil_lower_path).convert("L")
        shil_lower_mask = np.array(shil_lower_image) > 0

        # Load upper silhouette numpy array and convert to binary mask
        shil_upper_array = np.load(shil_upper_file)
        shil_upper_mask = np.array(shil_upper_array) > 0

        # Remove lower silhouette from upper silhouette
        corrected_upper_mask = np.logical_and(shil_upper_mask, ~shil_lower_mask[..., np.newaxis])
        corrected_upper_mask = corrected_upper_mask.squeeze(-1).astype(np.uint8) * 255

        # Save the corrected upper silhouette
        output_path = os.path.join(shil_upper_dir, shil_lower_file)
        Image.fromarray(corrected_upper_mask).save(output_path)


if __name__ == "__main__":
    # png_dir = '/hdd_data/nakul/soham/Dataset/Dress4D/processed/185/Take4/human/shils_lower'
    # npy_dir = '/hdd_data/nakul/soham/Dataset/Dress4D/processed/185/Take4/human/sam_seg'
    # output_dir = '/hdd_data/nakul/soham/Dataset/Dress4D/processed/185/Take4/human/shils_lower'
    # combine_segmentation_maps(png_dir, npy_dir, output_dir)

    ## Fix upper shil
    # shil_lower = '/hdd_data/nakul/soham/Dataset/Dress4D/processed/185/Take4/human/shils_lower'
    # shil_upper = '/hdd_data/nakul/soham/Dataset/Dress4D/processed/185/Take4/human/shils_upper'
    # fix_upper_shil(shil_lower, shil_upper)

    shil_lower = '/hdd_data/nakul/soham/Dataset/people_snapshot_public_proprecess/anran_tic/shils_lower'
    diffuse_global = '/hdd_data/nakul/soham/Dataset/people_snapshot_public_proprecess/anran_tic/diffuse'
    diffuse_lower = '/hdd_data/nakul/soham/Dataset/people_snapshot_public_proprecess/anran_tic/diffuse_lower'
    fix_diffuse_shil(shil_lower, diffuse_global, diffuse_lower)

