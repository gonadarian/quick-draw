import numpy as np
import libs.generators as gens


def ellipse_drawer(draw, dim, params):
    assert len(params) == 3
    x, y, a = params
    mid = (dim - 1) // 2
    draw.ellipse((x, y, dim - 1 - x, dim - 1 - y), fill=255)


def main():
    dim = 27
    center = dim // 2

    params = [5, 10, 0]
    image = gens.draw_image(dim, params, drawer=ellipse_drawer, show=True)

    images = []

    for x in range(center + 1):
        for y in range(dim):
            if x == center and y == center:
                break

            params = [x, y]
            image = gens.draw_image(dim, params, drawer=ellipse_drawer, show=False)
            images.append(image)

    images = np.array(images)
    print('shape:', images.shape)

    np.save('data\lines_27x27\line_27x27_centered_v1_{}x{}x{}.npy'.format(len(images), dim, dim), images)


if __name__ == '__main__':
    main()
    print('end')
