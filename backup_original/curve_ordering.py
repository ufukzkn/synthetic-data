import numpy as np



def order_curve_pixels_left_to_right(coords):
    S = set(map(tuple, coords))
    # start = leftmost pixel (min col), tie by row
    start = min(S, key=lambda p: (p[1], p[0]))
    path = [start]
    visited = {start}
    cur = start

    # neighbor offsets
    neigh = [(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0)]

    while True:
        candidates = []
        for dr,dc in neigh:
            nxt = (cur[0]+dr, cur[1]+dc)
            if nxt in S and nxt not in visited:
                candidates.append(nxt)
        if not candidates:
            break

        # Prefer increasing x (col), then minimal y change, then closeness
        cur_r, cur_c = cur
        candidates.sort(key=lambda p: (
            -(p[1]-cur_c),          # bigger dx preferred
            abs(p[0]-cur_r),        # smaller dy
            (p[0]-cur_r)**2 + (p[1]-cur_c)**2
        ))
        nxt = candidates[0]
        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return np.array(path)  # (K,2) rows, cols
