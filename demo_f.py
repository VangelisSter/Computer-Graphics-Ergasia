from Triangle_filling import render_img
import numpy as np
from pathlib import Path
from matplotlib import pylab as plt
import cv2

file_path = Path.cwd().parent / "hw1.npy"

data = np.load("hw1.npy", allow_pickle=True).item()

vertices = data['v_pos2d']     # Shape: (128776, 2)
faces = data['t_pos_idx']      # Shape: (63488, 3)
vcolors = data['v_clr']        # Shape: (128776, 3)
uvs = data['v_uvs']            # Shape: (128776, 2)
depth = data['depth']          # Shape: (128776,)


#vcolors = np.clip(vcolors, 0, 1)
dummy_texture = np.zeros((1, 1, 3), dtype=np.uint8)
print(f"Rendering {len(faces)} triangles. Please wait...")
flat_result = render_img(
    faces=faces, 
    vertices=vertices, 
    vcolors=vcolors, 
    uvs=uvs, 
    depth=depth, 
    shading='f',           # Trigger flat shading
    textImg=dummy_texture
)

if flat_result.dtype != np.uint8:
    save_img = np.clip(flat_result * 255, 0, 255).astype(np.uint8)
else:
    save_img = flat_result.copy()

save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

# Save the image to workspace
cv2.imwrite("my_flat_render.png", save_img_bgr)

print("Image successfully saved!")

plt.figure(figsize=(10, 10))
plt.imshow(flat_result)
plt.title("Final 3D Render: Flat Shading")
plt.axis('off') # Hide the axis ticks for a cleaner presentation
plt.show()