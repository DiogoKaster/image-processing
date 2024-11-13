import numpy as np
from scipy.signal import convolve2d

from image import Image
from utils.hysteresis_threshold import hysteresis_threshold


class GradientEdgeDetector:
    def __init__(
        self,
        image_data,
        filter_x=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        filter_y=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        low_threshold=0,
        high_threshold=0,
    ):
        self.filter_x = filter_x
        self.filter_y = filter_y
        self.image = image_data
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _apply_gradient(self):
        grad_x = convolve2d(self.image, self.filter_x, mode="same", boundary="wrap")
        grad_y = convolve2d(self.image, self.filter_y, mode="same", boundary="wrap")

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude

    def detect_edges(self):
        if len(self.image.shape) == 3:
            raise ValueError("Input image should be in grayscale")

        magnitude = self._apply_gradient()

        magnitude_normalized = np.uint8(255 * (magnitude / np.max(magnitude)))

        if self.low_threshold and self.high_threshold:
            magnitude_normalized = hysteresis_threshold(
                magnitude_normalized, self.low_threshold, self.high_threshold
            )

        return Image(data=magnitude_normalized)
