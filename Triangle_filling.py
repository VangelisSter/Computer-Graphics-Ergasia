import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def vector_interp(p1, p2, V1, V2, coord, dim):
    """
    Calculates the interpolated vector V at a specific coordinate.
    """
    #Since python uses 0 based indexing we have to reduce by 1
    axis_idx = dim - 1
    
    # Extract the coordinate values for the specified axis
    c1 = p1[axis_idx]
    c2 = p2[axis_idx]
    
    #If coordinates are the same, to avoid devision by zero return V1(could be V2)
    if c1 == c2:
        return np.copy(V1)
        
    # Calculate the linear interpolation weight w
    w = (coord - c1) / (c2 - c1)
    
    # Calculate the final interpolated vector V
    V = V1 + w * (V2 - V1)
    
    return V

def scanline_search(vertices, M, N):
    #In an MxN matrix, len() returns M, so we get the number of vertices
    K = len(vertices)
    edges = []

    xmin = []
    xmax = []
    for k in range(K):
        x1, y1 = vertices[k]
        x2, y2 = vertices[(k + 1) % K] #This matches v1 to v2, ..., v3 to v1 by circling amath.ceil using mod
        #We find xkmin etc
        ymin, ymax = int(min(y1, y2)), int(max(y1, y2))
        xmin.append(int(math.floor(min(x1, x2))))
        xmax.append(int(math.ceil(max(x1, x2))))

        #Since we want to exclude horizontal lines
        if ymin == ymax:
            continue

        #Reduce ymax by 1 to prevent double-counting pass-through vertices
        ymax = ymax - 1

        #Calculate the edge slope
        inv_m = (x2 - x1) / (y2 - y1) #We calculate the inverse to avoid division by zero without needing to check for it

        #Scanning from bottom to top
        x_of_ymin = x1 if y1 == ymin else x2

        #Store everyting needed about the edge in a dictionary
        edges.append({
            'k' : k, #The number of the edge
            'ymin' : ymin, #Its lowest point
            'ymax' : ymax, #Its highest point
            'inv_m' : inv_m, #Its slope
            'x_curr' : float(x_of_ymin) #The current x point(since we start from bottom to top, it's the x at ymin)
        })
    if len(edges) > 0:
        global_ymin = min(e['ymin'] for e in edges)
        global_ymax = max(e['ymax'] for e in edges)
        global_xmin = min(xmin)
        global_xmax = max(xmax)
    else:
        return [-1, -1]
    active_edges = [e for e in edges if e['ymin'] == global_ymin]

    for y in range(global_ymin, global_ymax + 1):

        #Sort the active edges from left to right
        active_edges.sort(key=lambda e: e['x_curr'])

        #The active intersections are just the x of the active edge at the current y
        active_intersections = [math.ceil(e['x_curr']) for e in active_edges]

        cross_count = 0

        if len(active_edges) >= 2:
            left_x = int(math.floor(active_edges[0]['x_curr']))
            right_x = int(math.ceil(active_edges[-1]['x_curr']))

        for x in range(left_x, right_x + 1):
        #for x in range(0, N + 1):

            #Count how many intersections we have for this particular x and add it to the count
            #cross_count +=  active_intersections.count(x)
            #cross parity check
            #if cross_count % 2 == 1:
            if 0 <= x < N and 0 <= y < M:
                yield (x, y)

        #Update recursively the active edge list

        #Exclude the edges if y_k,max = y, which means we have reached their end
        active_edges = [e for e in active_edges if e['ymax'] > y]

        #Add the edges if y_k,min = y + 1, which means we are about to scan them
        new_edges = [e for e in edges if e['ymin'] == y + 1]

        #Add the new edges to the active_edges list
        active_edges.extend(new_edges)

        #Update the current x position of each active edge by x + inv_m
        for e in active_edges:
            e['x_curr'] += e['inv_m']


def f_shading(img, vertices, vcolors):

    #We create a copy to not modify the original and to return it afterwards
    updated_img = np.copy(img)

    #First we need to get the dimenstions of the canvas, M,N
    M, N, _ = updated_img.shape

    #Now we need to calculate the flat color
    flat_color = np.mean(vcolors, axis=0)

    for x, y in scanline_search(vertices, M, N):
        if x >= 0 and y >= 0:
            updated_img[y, x] = flat_color
        #If the pixels coming from scanline search are valid, update them, else do nothing

    return updated_img

def barycentric_color(x, y, vertices, vcolors):
    x1, y1 = float(vertices[0][0]), float(vertices[0][1])
    x2, y2 = float(vertices[1][0]), float(vertices[1][1])
    x3, y3 = float(vertices[2][0]), float(vertices[2][1])

    #Calculate the total area A
    Area = ((y3 - y1) * (x2 - x1) - (x3 - x1) * (y2 - y1)) / 2

    #Calculate the small triangle area Ab
    Area_b = ((y - y3) * (x1 - x3) - (y1 - y3) * (x - x3)) / 2

    #Calculate the small triangle area Ac
    Area_c = ((y - y1) * (x2 - x1) - (y2 - y1) * (x - x1)) / 2

    #First check if the Area is 0
    if Area == 0:
        return vcolors[0]

    #Now the weights are
    w_b = Area_b / Area
    w_c = Area_c / Area

    #Since w_a + w_b + w_c = 1,
    w_a = 1.0 - w_b - w_c

    #Interpolate the final color
    final_color = w_b * vcolors[0] + w_a * vcolors[1] + w_c * vcolors[2]

    return np.clip(final_color, 0, 255)

def g_shading(img, vertices, vcolors):

    #We create a copy to not modify the original and to return it afterwards
    updated_img = np.copy(img)

    #First we need to get the dimenstions of the canvas, M,N
    M, N, _ = updated_img.shape

    for x, y in scanline_search(vertices, M, N):
        #To calculate the interpolated color of each pixel we will use the Barycentric method
        pixel_color = barycentric_color(x, y, vertices, vcolors)

        updated_img[y, x] = pixel_color
    return updated_img

def t_shading(img, vertices, uv, textImg):

    updated_img = np.copy(img)
    np_textImg = np.copy(textImg)
    M, N, _ = updated_img.shape
    K, L, _ = np_textImg.shape

    edges = []
    
    # Store all edge data, including vertices and UVs for vector_interp
    for k in range(3):
        p1 = vertices[k]
        p2 = vertices[(k + 1) % 3]
        uv1 = uv[k]
        uv2 = uv[(k + 1) % 3]
        
        y1, y2 = p1[1], p2[1]
        ymin, ymax = int(min(y1, y2)), int(max(y1, y2))
        
        if ymin == ymax: 
            continue # Skip horizontal edges
            
        ymax_adj = ymax - 1 
        inv_m = (p2[0] - p1[0]) / (p2[1] - p1[1])
        x_of_ymin = p1[0] if y1 == ymin else p2[0]
        
        edges.append({
            'ymin': ymin,
            'ymax': ymax_adj,
            'inv_m': inv_m,
            'x_curr': float(x_of_ymin),
            'p1': p1,
            'p2': p2, 
            'uv1': uv1,
            'uv2': uv2
        })
        
    if not edges:
        return updated_img
        
    global_ymin = min(e['ymin'] for e in edges)
    global_ymax = max(e['ymax'] for e in edges)
    active_edges = [e for e in edges if e['ymin'] == global_ymin]
    
    for y in range(global_ymin, global_ymax + 1):
        # Sort active edges left-to-right
        active_edges.sort(key=lambda e: e['x_curr'])
        
        # A triangle will have exactly 2 active edges (left and right boundary)
        if len(active_edges) >= 2:
            e_left = active_edges[0]
            e_right = active_edges[-1] # -1 ensures we grab the rightmost if they overlap
            
            xA = e_left['x_curr']
            xB = e_right['x_curr']
            
            # Interpolate UV at edge intersections A and B
            # dim=2 corresponds to the Y-axis in your vector_interp function
            uvA = vector_interp(e_left['p1'], e_left['p2'], e_left['uv1'], e_left['uv2'], y, 2)
            uvB = vector_interp(e_right['p1'], e_right['p2'], e_right['uv1'], e_right['uv2'], y, 2)
            
            # Define exact integer start and end for the X loop
            start_x = max(0, int(math.ceil(xA)))
            end_x = min(N - 1, int(math.ceil(xB)))
            
            # Interpolate UV across the scanline from A to B
            for x in range(start_x, end_x + 1):
                # dim=1 corresponds to the X-axis in your vector_interp function
                uvP = vector_interp([xA, y], [xB, y], uvA, uvB, x, 1)
                
                u, v = uvP[0], uvP[1]
                
                # Nearest Neighbor calculation as per the assignment text
                # We use K-1 and L-1 to prevent Index Out of Bounds errors
                tex_u = int(np.clip(math.ceil(u * (K - 1)), 0, K - 1))
                tex_v = int(np.clip(math.ceil(v * (L - 1)), 0, L - 1))
                
                if 0 <= y < M:
                    updated_img[y, x] = textImg[tex_u, tex_v]
        
        # Update Active Edges
        active_edges = [e for e in active_edges if e['ymax'] > y]
        new_edges = [e for e in edges if e['ymin'] == y + 1]
        active_edges.extend(new_edges)
        for e in active_edges:
            e['x_curr'] += e['inv_m']
            
    return updated_img

def render_img(faces, vertices, vcolors, uvs, depth, shading, textImg):

    #We are given
    M = 512
    N = 512
    # A white canvas
    img = np.ones((M, N, 3), dtype=np.uint8) * 255

    # Now we need to calculate the depth of each triangle
    triangles = []

    for face in faces:
        # Take the indices of the vertices
        idx0, idx1, idx2 = face

        # Get the depth of the vertices
        di = [depth[idx0], depth[idx1], depth[idx2]]

        # Calculate triangle depth (CoG)
        triangle_depth = np.mean(di)

        # Store the info to be used later
        triangles.append({
            'depth' : triangle_depth,
            'face_indices' : face
        })

    # Since we will color triangles from largest to smallest, we need to sort them in descending order
    triangles.sort(key=lambda t: t['depth'], reverse=True)

    # Render triangles in the sorted order
    for tri in tqdm(triangles, desc=f"Rendering ({shading} shading)"):
        idx0, idx1, idx2 = tri['face_indices']

        #Extract the specific data for the vertices
        tri_vertices = [vertices[idx0], vertices[idx1], vertices[idx2]]
        tri_colors = [vcolors[idx0], vcolors[idx1], vcolors[idx2]]
        tri_uvs = [uvs[idx0], uvs[idx1], uvs[idx2]]

        # Use the shading method given

        if shading in ['flat', 'f']:
            img = f_shading(img, tri_vertices, tri_colors)
        elif shading in ['gouraud', 'g']:
            img = g_shading(img, tri_vertices, tri_colors)
        elif shading in ['tex', 't']:
            img = t_shading(img, tri_vertices, tri_uvs, textImg)

    return img