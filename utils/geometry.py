import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import cKDTree

def barycentric_A(events, tri_obj):
    """
    Constructs the observation matrix A using barycentric interpolation.
    Maps points (events) to mesh nodes.
    """
    coords = tri_obj.points
    simplices = tri_obj.simplices
    n = coords.shape[0]
    M = events.shape[0]

    which = tri_obj.find_simplex(events)
    A = lil_matrix((M, n))
    inside = np.where(which >= 0)[0]

    # Inside simplices
    if inside.size > 0:
        T = tri_obj.transform[which[inside], :2]
        r = events[inside] - tri_obj.transform[which[inside], 2]
        bary12 = np.einsum("ijk,ik->ij", T, r)
        b0, b1 = bary12[:,0], bary12[:,1]
        b2 = 1 - b0 - b1
        verts = simplices[which[inside]]
        for i, (v0, v1, v2) in enumerate(verts):
            A[inside[i], v0] = b0[i]
            A[inside[i], v1] = b1[i]
            A[inside[i], v2] = b2[i]

    # Outside: nearest node
    outside = np.where(which < 0)[0]
    if outside.size > 0:
        tree = cKDTree(coords)
        _, idx = tree.query(events[outside])
        A[outside, idx] = 1.0

    return A.tocsr()