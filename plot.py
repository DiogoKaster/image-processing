import numpy as np
from matplotlib import pyplot as plt


class Plot:

    @staticmethod
    def histogram(
        hist, title="Grayscale Histogram", xLabel="Bins", yLabel="# of Pixels"
    ):
        hist = hist.flatten()
        hist_normalized = hist / np.sum(hist)

        plt.plot(hist_normalized)
        plt.xlim([0, 256])
        plt.ylim([0, max(hist_normalized)])
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

    @staticmethod
    def rgbHistogram(
        histogram, title="RGB Histogram", xLabel="Bins", yLabel="# of Pixels"
    ):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        # Plot histograms for each color channel
        colors = ["r", "g", "b"]
        for color in colors:
            plt.plot(histogram[color], color=color)

        plt.xlim([0, 256])
        plt.legend(["Red", "Green", "Blue"])
        plt.show()

    @staticmethod
    def transformation_function(l1, l2, k1, k2):
        # Definindo os limites de x
        x = np.linspace(0, 255, 256)

        # Inicializa a função de transformação
        y = np.zeros_like(x)

        # Define a transformação constante até l1
        y[x < l1] = k1

        # Define a transformação linear entre l1 e l2
        mask = (l1 <= x) & (x < l2)
        y[mask] = ((k2 - k1) / (l2 - l1)) * (x[mask] - l1) + k1

        # Define a transformação constante após l2
        y[x >= l2] = k2

        # Plota a transformação
        plt.plot(x, y, label="Transformation Function")
        plt.title("Piecewise Transformation Function")
        plt.xlabel("Original Pixel Intensity")
        plt.ylabel("Transformed Pixel Intensity")
        plt.axvline(x=l1, color="r", linestyle="--", label="l1")
        plt.axvline(x=l2, color="g", linestyle="--", label="l2")
        plt.axhline(y=k1, color="b", linestyle="--", label="k1")
        plt.axhline(y=k2, color="m", linestyle="--", label="k2")
        plt.legend()
        plt.grid()
        plt.xlim([0, 255])
        plt.ylim([0, 255])
        plt.show()

    @staticmethod
    def showImg(img, title="Display window"):
        # Exibe a imagem com matplotlib
        plt.figure(figsize=(8, 8))  # Ajusta o tamanho da figura
        plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        plt.title(title)
        plt.xlabel("Pixels - eixo X")
        plt.ylabel("Pixels - eixo Y")

        # Marca os pixels nos eixos para facilitar a visualização
        plt.xticks(range(0, img.shape[1], max(1, img.shape[1] // 10)))
        plt.yticks(range(0, img.shape[0], max(1, img.shape[0] // 10)))

        # Exibe uma barra de intensidade para imagens em escala de cinza
        if len(img.shape) == 2:
            plt.colorbar()

        plt.show()
