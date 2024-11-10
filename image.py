import cv2 as cv
import numpy as np


class Image:
    def __init__(self, path=None, data=None, rgb=False):
        if data is not None:
            self.data = data
            self.isRGB = rgb
        elif path:
            self.data = (
                cv.imread(path, cv.IMREAD_GRAYSCALE)
                if not rgb
                else cv.imread(path, cv.COLOR_BGR2RGB)
            )
            self.isRGB = rgb
        else:
            raise ValueError("Either path or data must be provided.")

        self.histogram = self.__calculateHistogram(self.data)

    def downsampling(self, xFactor=2, yFactor=2):
        height, width = self.data.shape[:2]
        new_height = int(height / yFactor)
        new_width = int(width / xFactor)

        new_data = np.zeros((new_height, new_width), dtype=self.data.dtype)

        row_scale = height / new_height
        col_scale = width / new_width

        for i in range(new_height):
            for j in range(new_width):
                original_row = int(i * row_scale)
                original_col = int(j * col_scale)
                new_data[i, j] = self.data[original_row, original_col]

        return Image(data=new_data, rgb=self.isRGB)

    def upsampling(self, xFactor=2, yFactor=2):
        height, width = self.data.shape[:2]
        new_height = height * yFactor
        new_width = width * xFactor

        new_data = np.zeros((new_height, new_width), dtype=self.data.dtype)

        row_scale = height / new_height
        col_scale = width / new_width

        for i in range(new_height):
            for j in range(new_width):
                original_row = int(i * row_scale)
                original_col = int(j * col_scale)
                new_data[i, j] = self.data[original_row, original_col]

        return Image(data=new_data, rgb=self.isRGB)

    def upsamplingNN(self, new_width=640, new_height=480):
        height, width = self.data.shape[:2]

        x_scale = new_width / width
        y_scale = new_height / height

        new_data = np.zeros((new_height, new_width), dtype=self.data.dtype)

        for i in range(new_height):
            for j in range(new_width):
                original_x = round(i / y_scale)
                original_y = round(j / x_scale)

                original_x = min(original_x, height - 1)
                original_y = min(original_y, width - 1)

                new_data[i, j] = self.data[original_x, original_y]

        return Image(data=new_data, rgb=self.isRGB)

    def upsamplingBI(self, new_width=2, new_height=2):
        height, width = self.data.shape[:2]

        x_scale = new_width / width
        y_scale = new_height / height

        new_data = np.zeros((new_height, new_width), dtype=self.data.dtype)

        for i in range(new_height):
            for j in range(new_width):
                src_x = i / y_scale
                src_y = j / x_scale

                x_floor = int(np.floor(src_x))
                y_floor = int(np.floor(src_y))
                x_ceil = min(int(np.ceil(src_x)), height - 1)
                y_ceil = min(int(np.ceil(src_y)), width - 1)

                W = src_x - x_floor
                H = src_y - y_floor

                I11 = self.data[x_floor, y_floor]
                I12 = self.data[x_floor, y_ceil]
                I21 = self.data[x_ceil, y_floor]
                I22 = self.data[x_ceil, y_ceil]

                new_data[i, j] = (
                    (1 - W) * (1 - H) * I11
                    + W * (1 - H) * I21
                    + (1 - W) * H * I12
                    + W * H * I22
                )

        return Image(data=new_data, rgb=self.isRGB)

    def piecewiseEqualization(self, l1=0, l2=255, k1=0, k2=255):
        channels = ["r", "g", "b"] if self.isRGB else [None]

        for ch in channels:
            hist = self.histogram[ch] if self.isRGB else self.histogram

            pdf = hist / hist.sum()
            cdf = pdf.cumsum()
            cdf_normalized = np.floor(
                (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            ).astype(np.uint8)

            channel_data = (
                self.data[:, :, channels.index(ch)].copy() if ch else self.data.copy()
            )
            for row in range(channel_data.shape[0]):
                for col in range(channel_data.shape[1]):
                    original_pixel = channel_data[row, col]
                    cdf_value = cdf_normalized[original_pixel]
                    channel_data[row, col] = round(
                        self.__piecewise_transform(cdf_value, l1, l2, k1, k2)
                    )

        return Image(data=channel_data, rgb=self.isRGB)

    def __piecewise_transform(self, cdf_value, l1, l2, k1, k2):
        if cdf_value < l1:
            return k1
        elif l1 <= cdf_value < l2:
            return ((k2 - k1) / (l2 - l1)) * (cdf_value - l1) + k1
        else:
            return k2

    def __calculateHistogram(self, data):
        if len(data.shape) == 2:
            return np.bincount(data.ravel(), minlength=256)
        else:
            return {
                "r": np.bincount(data[:, :, 0].ravel(), minlength=256),
                "g": np.bincount(data[:, :, 1].ravel(), minlength=256),
                "b": np.bincount(data[:, :, 2].ravel(), minlength=256),
            }

    def showImg(self):
        cv.imshow("Display window", self.data)
        cv.waitKey(0)
