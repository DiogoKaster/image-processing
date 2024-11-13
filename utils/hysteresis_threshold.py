import numpy as np


def hysteresis_threshold(magnitude, low_threshold, high_threshold):
    edges = np.zeros_like(magnitude, dtype=np.uint8)

    strong_edges = magnitude >= high_threshold
    edges[strong_edges] = 255

    weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)

    weak_edges_dilated = np.zeros_like(magnitude, dtype=bool)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i - 1 : i + 2, j - 1 : j + 2]):
                    weak_edges_dilated[i, j] = True

    edges[weak_edges_dilated] = 255

    return edges
