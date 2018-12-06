import numpy as np
import libs.generators as gens


def line_drawer(draw, dim, params):
    assert len(params) == 2
    x, y = params
    draw.line((x, y, dim - 1 - x, dim - 1 - y), fill=255)


def main():
    dim = 28
    image_list = []

    for x in range(dim // 2):
        for y in range(dim):
            params = [x, y]
            image = gens.draw_image(dim, params, drawer=line_drawer, show=False)
            image_list.append(image)

    image_list = np.array(image_list)
    print('shape:', image_list.shape)

    np.save('data\line_originals_v2_392x28x28.npy', image_list)


if __name__ == '__main__':
    main()
    print('end')
