from collections import defaultdict
import numpy as np
from skimage.measure import label, regionprops

def connected_components_skeleton(skel01):
    # skel01: 0/1 uint8
    lab = label(skel01 > 0, connectivity=2)
    comps = []
    for r in regionprops(lab):
        coords = r.coords  # (row, col)
        comps.append(coords)
    return comps

def find_endpoints(coords_set):
    # endpoint = skeleton pixel with exactly one neighbor in 8-connectivity
    coords = np.array(list(coords_set))
    S = coords_set
    endpoints = []
    for (r,c) in coords:
        n = 0
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: 
                    continue
                if (r+dr, c+dc) in S:
                    n += 1
        if n == 1:
            endpoints.append((r,c))
    return endpoints

def stitch_fragments(fragments, max_y_gap=6, max_x_back=10):
    """
    fragments: list of np arrays (N,2) [row,col] each consider as a piece
    returns: list of tracks; each track is list of fragments indices
    """
    # Build fragment meta
    meta = []
    for i, coords in enumerate(fragments):
        S = set(map(tuple, coords))
        ends = find_endpoints(S)
        cols = coords[:,1]
        rows = coords[:,0]
        meta.append({
            "i": i,
            "minx": int(cols.min()),
            "maxx": int(cols.max()),
            "meanx": float(cols.mean()),
            "meany": float(rows.mean()),
            "ends": ends,  # may be 0,2, or more if noisy
            "coords": coords
        })

    # Sort fragments left->right by minx
    order = sorted(meta, key=lambda m: m["minx"])
    used = set()
    tracks = []

    def rightmost_endpoint(m):
        if not m["ends"]:
            # fallback: take point with max x
            coords = m["coords"]
            j = np.argmax(coords[:,1])
            return tuple(coords[j])
        return max(m["ends"], key=lambda p: p[1])  # max col

    def leftmost_endpoint(m):
        if not m["ends"]:
            coords = m["coords"]
            j = np.argmin(coords[:,1])
            return tuple(coords[j])
        return min(m["ends"], key=lambda p: p[1])  # min col

    for m in order:
        if m["i"] in used:
            continue
        track = [m["i"]]
        used.add(m["i"])
        cur = m

        while True:
            cur_r_end = rightmost_endpoint(cur)
            best = None
            best_score = 1e9

            for cand in order:
                if cand["i"] in used:
                    continue
                # must be to the right (allow tiny backwards due to noise)
                if cand["minx"] < cur["maxx"] - max_x_back:
                    continue

                cand_l_end = leftmost_endpoint(cand)

                dx = cand_l_end[1] - cur_r_end[1]
                dy = abs(cand_l_end[0] - cur_r_end[0])

                if dx < -max_x_back:
                    continue
                if dy > max_y_gap:
                    continue

                score = (max(dx, 0)*0.5) + dy
                if score < best_score:
                    best_score = score
                    best = cand

            if best is None:
                break

            track.append(best["i"])
            used.add(best["i"])
            cur = best

        tracks.append(track)

    return tracks
