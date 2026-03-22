from Triangle_filling import render_img
import numpy as np
from pathlib import Path
from matplotlib import pylab as plt

file_path = Path.cwd().parent / "hw1.npy"

data = np.load("hw1.npy", allow_pickle=True).item()

texture_path = "loony-repeat.png"
texture_img = plt.imread(texture_path)

# CRITICAL STEP: Matplotlib loads PNGs as floating point numbers (0.0 to 1.0).
# Your canvas expects integers from 0 to 255. We must convert it here.
if texture_img.dtype == np.float32 or texture_img.dtype == np.float64:
    texture_img = (texture_img * 255).astype(np.uint8)

# Strip the alpha (transparency) channel if the PNG has one, keeping only RGB
if texture_img.shape[-1] == 4:
    texture_img = texture_img[..., :3]


vertices = data['v_pos2d']     # Shape: (128776, 2)
faces = data['t_pos_idx']      # Shape: (63488, 3)
vcolors = data['v_clr']        # Shape: (128776, 3)
uvs = data['v_uvs']            # Shape: (128776, 2)
depth = data['depth']          # Shape: (128776,)


vcolors = np.clip(vcolors * 255, 0, 255).astype(np.uint8)
print(f"Rendering {len(faces)} triangles. Please wait...")
texture_result = render_img(
    faces=faces, 
    vertices=vertices, 
    vcolors=vcolors, 
    uvs=uvs, 
    depth=depth, 
    shading='t',           # Trigger texture shading
    textImg=texture_img
)

plt.figure(figsize=(10, 10))
plt.imshow(texture_result)
plt.title("Final 3D Render: Texture Shading")
# Invert Y-axis if the image appears upside down (standard in many 2D graphics systems)
# plt.gca().invert_yaxis() 
plt.axis('off') # Hide the axis ticks for a cleaner presentation
plt.show()