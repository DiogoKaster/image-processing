import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from image import Image
from utils.hysteresis_threshold import hysteresis_threshold


class CannyEdgeDetector:
    def __init__(self, image_data, low_threshold=30, high_threshold=80, sigma=0.7):
        self.image = image_data
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sigma = sigma

    def _apply_gaussian_blur(self):
        return gaussian_filter(self.image, sigma=self.sigma)

    def _compute_gradient_magnitude_and_direction(self, smoothed_image):
        filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = convolve2d(smoothed_image, filter_x, mode="same", boundary="wrap")
        grad_y = convolve2d(smoothed_image, filter_y, mode="same", boundary="wrap")

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        return magnitude, direction

    def _non_maximum_suppression(self, magnitude, direction):
        suppressed = np.zeros(magnitude.shape, dtype=np.float32)
        angle = np.rad2deg(direction) % 180

        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed

    def detect_edges(self):
        if len(self.image.shape) == 3:
            raise ValueError("Input image should be in grayscale")

        smoothed_image = self._apply_gaussian_blur()

        magnitude, direction = self._compute_gradient_magnitude_and_direction(
            smoothed_image
        )

        suppressed = self._non_maximum_suppression(magnitude, direction)

        edges = hysteresis_threshold(
            suppressed, self.low_threshold, self.high_threshold
        )

        return Image(data=edges)
