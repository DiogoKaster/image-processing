import numpy as np
from scipy.signal import convolve2d

from image import Image
from utils.hysteresis_threshold import hysteresis_threshold


class LaplacianEdgeDetector:
    def __init__(
        self,
        image_data,
        filter_laplace=np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]]),
        low_threshold=0,
        high_threshold=0,
    ):
        self.filter_laplace = filter_laplace
        self.image = image_data
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _apply_laplacian(self):
        laplacian = convolve2d(
            self.image, self.filter_laplace, mode="same", boundary="wrap"
        )
        magnitude = np.abs(laplacian)
        return magnitude

    def detect_edges(self):
        if len(self.image.shape) == 3:
            raise ValueError("Input image should be in grayscale")

        magnitude = self._apply_laplacian()

        magnitude_normalized = np.uint8(255 * (magnitude / np.max(magnitude)))

        if self.low_threshold and self.high_threshold:
            magnitude_normalized = hysteresis_threshold(
                magnitude_normalized, self.low_threshold, self.high_threshold
            )

        return Image(data=magnitude_normalized)
