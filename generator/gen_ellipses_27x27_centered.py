import numpy as np
import libs.generators as gens


def ellipse_drawer(draw, dim, params):
    assert len(params) == 2
    x, y = params
    draw.ellipse((x, y, dim - 1 - x, dim - 1 - y), outline=255)


def main():
    dim = 27
    center = dim // 2 - 1

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

    np.save('data\ellipse\ellipse_27x27_centered_v1_{}x{}x{}.npy'.format(len(images), dim, dim), images)


if __name__ == '__main__':
    main()
    print('end')
