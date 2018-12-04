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
    center = dim // 2
    images = []

    for x in range(center + 1):
        for y in range(dim):
            if x == center and y == center:
                break
            image = get_img_array(x, y, show=False)
            images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    np.save('data\lines_27x27\line_27x27_centered_v1_{}x{}x{}.npy'.format(len(images), dim, dim), images)


if __name__ == '__main__':
    main()
    print('end')
