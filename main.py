from image import Image
from plot import Plot

img = Image("images/checker.png")
Plot.showImg(img.data)
new_img = img.upsamplingBI()
Plot.showImg(new_img.data)
new_img2 = new_img.upsamplingBI(new_width=4, new_height=4)
Plot.showImg(new_img2.data)
Plot.showImg(new_img2.upsamplingBI(new_width=8, new_height=8).data)
