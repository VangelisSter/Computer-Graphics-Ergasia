import numpy as np
import matplotlib.pyplot as plt

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
        x2, y2 = vertices[(k + 1) % K] #This matches v1 to v2, ..., v3 to v1 by circling around using mod

        #We find xkmin etc
        ymin, ymax = int(min(y1, y2)), int(max(y1, y2))
        xmin.append(int(min(x1, x2)))
        xmax.append(int(max(x1, x2)))

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

    global_ymin = min(e['ymin'] for e in edges)
    global_ymax = max(e['ymax'] for e in edges)
    global_xmin = min(xmin)
    global_xmax = max(xmax)

    active_edges = [e for e in edges if e['ymin'] == global_ymin]

    for y in range(global_ymin, global_ymax + 1):

        #Sort the active edges from left to right
        active_edges.sort(key=lambda e: e['x_curr'])

        #The active intersections are just the x of the active edge at the current y
        active_intersections = [round(e['x_curr']) for e in active_edges]

        cross_count = 0
        for x in range(global_xmin -1, global_xmax + 2):
        #for x in range(0, N + 1):

            #Count how many intersections we have for this particular x and add it to the count
            cross_count +=  active_intersections.count(x)

            #cross parity check
            if cross_count % 2 == 1:
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
        updated_img[x, y] = flat_color

    return updated_img


# --- Test Execution ---

# 1. Create a black canvas (200x200 pixels)
canvas = np.zeros((200, 200, 3), dtype=np.uint8)

# 2. Define vertices (x, y)
# Triangle shape
test_vertices = [(50, 50), (150, 80), (150, 160)]

# 3. Define colors for each vertex (RGB)
# Red, Green, Blue -> The mean will be a greyish tone
test_colors = np.array([
    [255, 255, 0], # Yellow
    [255, 255, 0], # Yellow
    [255, 255, 0]  # Yellow
])
# 4. Run the function
result_img = f_shading(canvas, test_vertices, test_colors)

# 5. Display the result
plt.imshow(result_img)
plt.title("Flat Shading Test (Scanline)")
plt.gca().invert_yaxis() # Match the y-coordinate logic if necessary
plt.show()