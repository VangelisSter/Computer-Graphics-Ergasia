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

def f_shading(img, vertices, vcolors):

    #We create a copy to not modify the original and to return it afterwards
    updated_img = np.copy(img)

    #First we need to get the dimenstions of the canvas, M,N
    M, N, _ = updated_img.shape

    #Now we need to calculate the flat color
    flat_color = np.mean(vcolors, axis=0)

    v1 = vertices[0,:]
    v2 = vertices[1,:]
    v3 = vertices[2,:]

    edges = [v2 - v1, v3 - v2, v1 - v3]

    xkmin = [min(v1[0], v2[0]), min(v2[0], v3[0]), min(v1[0], v3[0])]
    ykmin = [min(v1[1], v2[1]), min(v2[1], v3[1]), min(v1[1], v3[1])]
    xkmax = [max(v1[0], v2[0]), max(v2[0], v3[0]), max(v1[0], v3[0])]
    ykmax = [max(v1[1], v2[1]), max(v2[1], v3[1]), max(v1[1], v3[1])]
    mk = [(v2[1] - v1[1])/(v2[0]- v1[0]), (v3[1] - v2[1])/(v3[0]- v2[0]), (v1[1] - v3[1])/(v1[0]- v3[0])]
    ymin = min(ykmin)
    min_index = ykmin.index(ymin)
    ymax = max(ykmax)

    active_edges = [min_index]
    active_marginal_points = [] 

    for y in range(ymin, ymax):
        active_marginal_points
        cross_count = 0
        for x in range(0, N):
            if x in active_marginal_points[:, 0]:
                cross_count = cross_count + 1
            if cross_count % 2 == 1:
                updated_img[x][y] = flat_color
    return updated_img