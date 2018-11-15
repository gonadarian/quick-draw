from PIL import Image, ImageDraw
import numpy as np


img_size = 28


def get_img_array(x, y, show=False):
    im = Image.new("L", (28, 28), "black")

    draw = ImageDraw.Draw(im)
    draw.line((x, y, img_size-1 - x, img_size-1 - y), fill=255)
    im_array = np.asarray(im)
    if show:
        print(im_array)
        # im.show()

    return im_array


# get_img_array(5, 10, True)

im_set = np.empty([img_size ** 2 // 2, img_size, img_size])

for x in range(img_size // 2):
    for y in range(img_size):
        im_array = get_img_array(x, y, False)
        im_set[x * img_size + y, :, :] = im_array

np.save('data\line_originals_v2_392x28x28.npy', im_set)


print('end')
