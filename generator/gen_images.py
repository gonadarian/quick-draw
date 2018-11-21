from PIL import Image, ImageDraw
import numpy as np


img_size = 28


def get_img_array(x, y, show=False):
    image = Image.new("L", (28, 28), "black")

    draw = ImageDraw.Draw(image)
    draw.line((x, y, img_size-1 - x, img_size-1 - y), fill=255)
    image_array = np.asarray(image)

    if show:
        print(image_array)
        image.show()

    return image_array


def main():

    im_set = np.empty([img_size ** 2 // 2, img_size, img_size])

    for x in range(img_size // 2):
        for y in range(img_size):
            im_array = get_img_array(x, y, False)
            im_set[x * img_size + y, :, :] = im_array

    np.save('data\line_originals_v2_392x28x28.npy', im_set)


if __name__ == '__main__':
    main()
    print('end')
