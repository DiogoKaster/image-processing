from algorithms.edge_detection.canny_edge import CannyEdgeDetector
from algorithms.edge_detection.gradient_edge_operator import GradientEdgeDetector
from image import Image
from plot import Plot

img = Image("images/lena.png")
Plot.showImg(img.data)
new_img = CannyEdgeDetector(image_data=img.data).detect_edges()
Plot.showImg(new_img.data)
