from PIL import Image, ImageDraw
import numpy as np


dim = 27


def get_img_array(x, y, show=False):
    image = Image.new("L", (dim, dim), "black")

    draw = ImageDraw.Draw(image)
    draw.line((x, y, dim - 1 - x, dim - 1 - y), fill=255)
    image_array = np.asarray(image)

    if show:
        print(image_array)
        image.show()

    return image_array


def main():
    images = []

    for x in range(14):
        for y in range(27):
            if x == 13 and y == 13:
                break
            image = get_img_array(x, y, show=False)
            images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    np.save('data\lines_27x27_centered_v1_{}x{}x{}.npy'.format(len(images), dim, dim), im_set)


if __name__ == '__main__':
    main()
    print('end')
