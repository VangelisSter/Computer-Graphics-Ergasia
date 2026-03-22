from Triangle_filling import render_img
import numpy as np
from pathlib import Path
from matplotlib import pylab as plt
import cv2

file_path = Path.cwd().parent / "hw1.npy"

data = np.load("hw1.npy", allow_pickle=True).item()

texture_path = "loony-repeat.png"
texture_img = plt.imread(texture_path)

# Matplotlib loads PNGs as floating point numbers (0.0 to 1.0).
# Canvas expects integers from 0 to 255. We must convert it .
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


if texture_result.dtype != np.uint8:
    save_img = np.clip(texture_result * 255, 0, 255).astype(np.uint8)
else:
    save_img = texture_result.copy()

save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

# Save the image to workspace
cv2.imwrite("my_texture_render.png", save_img_bgr)

print("Image successfully saved!")
plt.figure(figsize=(10, 10))
plt.imshow(texture_result)
plt.title("Final 3D Render: Texture Shading")
plt.axis('off') # Hide the axis ticks for a cleaner presentation
plt.show()