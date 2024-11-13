import numpy as np


def hysteresis_threshold(magnitude, low_threshold, high_threshold):
    edges = np.zeros_like(magnitude, dtype=np.uint8)

    strong_edges = magnitude >= high_threshold
    edges[strong_edges] = 255

    weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)

    for y in range(1, magnitude.shape[0] - 1):
        for x in range(1, magnitude.shape[1] - 1):
            if weak_edges[y, x]:
                if np.any(strong_edges[y - 1 : y + 2, x - 1 : x + 2]):
                    edges[y, x] = 255

    return edges
