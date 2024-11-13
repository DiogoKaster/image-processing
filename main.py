from algorithms.edge_detection.gradient_edge_detector import GradientEdgeDetector
from algorithms.edge_detection.laplace_edge_detector import LaplacianEdgeDetector
from image import Image
from plot import Plot

img = Image("images/lena.png")
new_img = GradientEdgeDetector(image_data=img.data).detect_edges()
Plot.showImg(new_img.data)
